"""
train_evals.py - sharpen an existing net's value head using Stockfish
centipawn evaluations instead of game results.

Usage:
    python train_evals.py --data data/evals.bin
    python train_evals.py --data data/evals.bin --init checkpoints/finetuned_best.pt
    python train_evals.py --data data/evals.bin --value-weight 4 --steps 0

    # train a brand new, larger net from random weights
    python train_evals.py --data data/evals.bin --scratch \
        --blocks 10 --channels 192 --value-weight 1 --steps 0 \
        --out checkpoints/v2

Options:
    --data          eval records from extract_evals.py   (required)
    --scratch       start from random weights instead of a checkpoint
    --blocks        residual blocks, only with --scratch        (default 6)
    --channels      trunk width, only with --scratch            (default 128)
    --init          weights to start from   (default checkpoints/finetuned_best.pt)
    --out           checkpoint prefix       (default checkpoints/evaltuned)
    --lr            learning rate     (default 1e-4, or 1e-3 with --scratch)
    --value-weight  how much the value head matters vs policy  (default 3.0)
    --scale         centipawns per unit of tanh input         (default 400)
    --steps         stop after this many steps, 0 = never     (default 30000)
    --sleep         idle seconds per step

Why the value weight defaults high: the policy head is already good, and
the whole point of this pass is the value head. Weighting value above
policy keeps the run focused on the thing that needs fixing.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from train import ChessNet, pick_device, PLANES, BATCH, LOG_EVERY, EVAL_EVERY, SAVE_EVERY
from extract_evals import EVAL_RECORD

VAL_POSITIONS = 100_000
EYE13 = np.eye(13, dtype=np.float32)


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def expand(recs, scale):
    """Same 18 planes as train.py, but the value target is continuous."""
    b = len(recs)

    pieces = recs["pieces"].astype(np.int64)
    planes = EYE13[pieces][:, :, 1:]
    planes = planes.transpose(0, 2, 1).reshape(b, 12, 8, 8)

    extra = np.zeros((b, 6, 8, 8), dtype=np.float32)
    castling = recs["castling"].astype(np.int32)
    for bit in range(4):
        extra[:, bit] = ((castling >> bit) & 1).astype(np.float32)[:, None, None]
    extra[:, 4] = recs["stm"].astype(np.float32)[:, None, None]

    ep = recs["ep"].astype(np.int32)
    ep_plane = np.zeros((b, 64), dtype=np.float32)
    rows = np.nonzero(ep >= 0)[0]
    ep_plane[rows, ep[rows]] = 1.0
    extra[:, 5] = ep_plane.reshape(b, 8, 8)

    x = np.concatenate([planes, extra], axis=1)

    move = recs["from_sq"].astype(np.int64) * 64 + recs["to_sq"].astype(np.int64)

    # Squash centipawns into the tanh range the value head produces. A larger
    # scale keeps more resolution at high advantages, which is exactly what
    # was missing when the net could not tell +7 from +12.
    value = np.tanh(recs["cp"].astype(np.float32) / scale)

    return x, move, value


def to_device(recs, device, scale):
    x, move, value = expand(recs, scale)
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(move).to(device),
            torch.from_numpy(value).to(device))


@torch.no_grad()
def assess(model, val, device, scale, batches=20):
    """Report policy match and value error separately - they move apart here."""
    model.eval()
    correct = seen = 0
    abs_err = 0.0
    for i in range(batches):
        recs = val[i * BATCH:(i + 1) * BATCH]
        if len(recs) == 0:
            break
        x, move, value = to_device(recs, device, scale)
        p, v = model(x)
        correct += (p.argmax(1) == move).sum().item()
        abs_err += (v - value).abs().sum().item()
        seen += len(recs)
    model.train()
    return correct / seen, abs_err / seen


def main():
    data_path = arg("--data")
    if data_path is None:
        print(__doc__)
        sys.exit(1)

    scratch = "--scratch" in sys.argv
    init_path = arg("--init", "checkpoints/finetuned_best.pt")
    out_prefix = arg("--out", "checkpoints/evaltuned")
    # Fine-tuning wants a gentle rate; a fresh net needs a much larger one.
    lr = arg("--lr", 1e-3 if scratch else 1e-4, float)
    value_weight = arg("--value-weight", 3.0, float)
    scale = arg("--scale", 400.0, float)
    max_steps = arg("--steps", 30_000, int)
    sleep_per_step = arg("--sleep", 0.0, float)

    latest_path = out_prefix + "_latest.pt"
    best_path = out_prefix + "_best.pt"

    device = pick_device()
    print("device:", device)

    data = np.memmap(data_path, dtype=EVAL_RECORD, mode="r")
    n_val = min(VAL_POSITIONS, len(data) // 100)
    train_data = data[:-n_val]
    val_data = data[-n_val:]
    print(f"train positions: {len(train_data):,}   validation: {len(val_data):,}")

    blocks = arg("--blocks", 6, int)
    channels = arg("--channels", 128, int)
    base_step = 0
    resuming = False

    if os.path.exists(latest_path):
        # An interrupted run always wins - its architecture is authoritative.
        ckpt = torch.load(latest_path, map_location=device)
        blocks = ckpt.get("blocks", blocks)
        channels = ckpt.get("channels", channels)
        model = ChessNet(blocks=blocks, channels=channels).to(device)
        model.load_state_dict(ckpt["model"])
        step = ckpt.get("eval_step", 0)
        best_err = ckpt.get("best_err", float("inf"))
        base_step = ckpt.get("step", 0)
        resuming = True
        source_desc = f"resumed from {latest_path}"
    elif scratch:
        ckpt = {}
        model = ChessNet(blocks=blocks, channels=channels).to(device)
        step = 0
        best_err = float("inf")
        source_desc = "fresh random weights"
    else:
        if not os.path.exists(init_path):
            print(f"no checkpoint at {init_path} "
                  f"(use --scratch to start from random weights)")
            sys.exit(1)
        ckpt = torch.load(init_path, map_location=device)
        blocks = ckpt.get("blocks", 6)
        channels = ckpt.get("channels", 128)
        model = ChessNet(blocks=blocks, channels=channels).to(device)
        model.load_state_dict(ckpt["model"])
        step = 0
        best_err = float("inf")
        base_step = ckpt.get("step", 0)
        source_desc = f"initialised from {init_path}"

    params = sum(p.numel() for p in model.parameters())
    print(source_desc)
    print(f"model: {blocks} blocks x {channels} channels, "
          f"{params/1e6:.1f}M parameters")
    print(f"lr {lr}, value weight {value_weight}, scale {scale}cp")
    if sleep_per_step:
        print(f"throttled: idling {sleep_per_step*1000:.0f} ms after each step")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if resuming and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])

    acc, err = assess(model, val_data, device, scale)
    print(f"\nbefore: policy match {acc*100:.1f}%, "
          f"mean value error {err:.4f}\n")

    def save(path, with_opt):
        blob = {"model": model.state_dict(),
                "step": base_step,
                "eval_step": step,
                "blocks": blocks,
                "channels": channels,
                "best_err": best_err}
        if with_opt:
            blob["opt"] = opt.state_dict()
        torch.save(blob, path)

    rng = np.random.default_rng()
    running_p = running_v = 0.0
    last_log = time.time()

    try:
        while not max_steps or step < max_steps:
            idx = np.sort(rng.integers(0, len(train_data), BATCH))
            x, move, value = to_device(train_data[idx], device, scale)

            p, v = model(x)
            policy_loss = F.cross_entropy(p, move)
            value_loss = F.mse_loss(v, value)
            loss = policy_loss + value_weight * value_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_p += policy_loss.item()
            running_v += value_loss.item()
            step += 1

            if sleep_per_step:
                time.sleep(sleep_per_step)

            if step % LOG_EVERY == 0:
                rate = LOG_EVERY / (time.time() - last_log)
                print(f"step {step:>7,} | policy {running_p/LOG_EVERY:5.3f} | "
                      f"value {running_v/LOG_EVERY:6.4f} | {rate:5.1f} steps/s")
                running_p = running_v = 0.0
                last_log = time.time()

            if step % EVAL_EVERY == 0:
                acc, err = assess(model, val_data, device, scale)
                marker = ""
                if err < best_err:
                    best_err = err
                    save(best_path, with_opt=False)
                    marker = "   <- new best, saved"
                print(f"    validation: policy {acc*100:.1f}%  "
                      f"value error {err:.4f}{marker}")
                last_log = time.time()

            if step % SAVE_EVERY == 0:
                save(latest_path, with_opt=True)

    except KeyboardInterrupt:
        print("\nstopping...")

    save(latest_path, with_opt=True)
    print(f"saved at eval step {step:,}  (best value error {best_err:.4f})")
    print(f"  {latest_path}\n  {best_path}")


if __name__ == "__main__":
    main()
