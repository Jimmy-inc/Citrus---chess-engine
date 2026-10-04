"""
train_lc0.py - train the version two network on Leela's visit distributions.

The difference from train_evals.py is the policy target. There, each position
came with one move and the loss asked "was this the move?". Here each position
comes with the share of search visits every move received, and the loss asks
"was this the whole distribution?" - which carries far more: that two moves
were nearly equal, that a third was considered and rejected, how sharp the
position was.

Concretely the loss becomes cross-entropy against soft targets,

    -sum_m  visits(m) * log p(m)

over the moves actually stored, rather than -log p(best move).

The value target needs no rescaling either. Leela stores it already in
[-1, 1] from the mover's point of view, which is exactly what a tanh head
produces - so there is no arbitrary centipawn scale in the way.

Usage:
    python train_lc0.py --data data/lc0.bin
    python train_lc0.py --data 'data/lc0_*.bin' --init checkpoints/v2_best.pt
    python train_lc0.py --data data/lc0.bin --value blend --lr 2e-4

Options:
    --data          converted records; a glob matches several files (required)
    --init          start from an existing v2 checkpoint rather than random
    --blocks        residual blocks, ignored when --init is given (default 12)
    --channels      trunk width, ignored when --init is given   (default 256)
    --out           checkpoint prefix                (default checkpoints/lc0)
    --lr            learning rate                              (default 5e-4)
    --value         target: result, best, or blend of both   (default blend)
    --value-weight  value loss against policy loss            (default 1.0)
    --batch         positions per step                          (default 512)
    --top           policy slots per record, must match convert   (default 30)
    --steps         stop after this many, 0 = never          (default 500000)
    --clip          gradient norm clip, 0 disables              (default 1.0)
    --fp32          disable mixed precision
"""

import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding_v2 import PLANES, POLICY_SIZE, planes_from_pieces
from train_v2 import ChessNetV2, pick_device, amp_dtype, device_supports_amp
from convert_lc0 import make_dtype

VAL_POSITIONS = 200_000
LOG_EVERY = 100
EVAL_EVERY = 2_000
SAVE_EVERY = 2_000


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def load_records(pattern, dtype):
    """Memory-map one file, or several matching a glob, as one array."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"nothing matches {pattern}")
        sys.exit(1)
    parts = [np.memmap(p, dtype=dtype, mode="r") for p in paths]
    print(f"{len(paths)} file(s), "
          f"{sum(len(p) for p in parts):,} records total")
    for path, part in zip(paths, parts):
        print(f"  {os.path.basename(path):<28} {len(part):>12,}")
    return parts


def draw(parts, count, rng):
    """Sample a batch across all the mapped files."""
    sizes = np.array([len(p) for p in parts], dtype=np.int64)
    which = rng.choice(len(parts), size=count, p=sizes / sizes.sum())
    out = []
    for index in range(len(parts)):
        rows = np.nonzero(which == index)[0]
        if len(rows) == 0:
            continue
        picks = np.sort(rng.integers(0, sizes[index], len(rows)))
        out.append(parts[index][picks])
    return np.concatenate(out) if len(out) > 1 else out[0]


def expand(recs, value_mode):
    """Planes, sparse policy targets and value targets for a batch."""
    count = len(recs)
    pieces = recs["pieces"]
    castling = recs["castling"]
    ep = np.full(count, -1, dtype=np.int8)

    # Leela's planes are already canonical, so nothing needs flipping here.
    x = planes_from_pieces(pieces, castling, ep)

    idx = recs["idx"].astype(np.int64)
    prob = recs["prob"].astype(np.float32)

    if value_mode == "result":
        value = recs["result_q"].astype(np.float32)
    elif value_mode == "best":
        value = recs["best_q"].astype(np.float32)
    else:
        # Leela trains against both: the game's actual outcome is unbiased
        # but noisy, the search evaluation is smoother but can inherit the
        # network's own mistakes. Half each is the usual compromise.
        value = 0.5 * (recs["result_q"].astype(np.float32)
                       + recs["best_q"].astype(np.float32))

    return x, idx, prob, np.clip(value, -1.0, 1.0)


def to_device(recs, device, value_mode):
    x, idx, prob, value = expand(recs, value_mode)
    return (torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(idx).to(device, non_blocking=True),
            torch.from_numpy(prob).to(device, non_blocking=True),
            torch.from_numpy(value).to(device, non_blocking=True))


def policy_loss(logits, idx, prob):
    """
    Cross-entropy against the visit distribution.

    Only the stored moves matter; padding slots carry probability zero and
    so contribute nothing, which is why they need no explicit mask.
    """
    log_p = F.log_softmax(logits.float(), dim=1)
    picked = log_p.gather(1, idx)
    return -(prob * picked).sum(dim=1).mean()


@torch.no_grad()
def assess(model, val, device, value_mode, batch, batches=20):
    model.eval()
    seen = 0
    top1 = 0
    policy_total = value_total = 0.0
    for i in range(batches):
        recs = val[i * batch:(i + 1) * batch]
        if len(recs) == 0:
            break
        x, idx, prob, value = to_device(recs, device, value_mode)
        with torch.autocast("cuda", dtype=amp_dtype(),
                            enabled=torch.cuda.is_available()):
            logits, predicted = model(x)
        predicted = predicted.float()

        policy_total += policy_loss(logits, idx, prob).item() * len(recs)
        value_total += F.mse_loss(predicted, value).item() * len(recs)
        # Does the net's favourite move match the most-visited one?
        top1 += (logits.argmax(1) == idx[:, 0]).sum().item()
        seen += len(recs)

    model.train()
    return policy_total / seen, value_total / seen, top1 / seen


def main():
    pattern = arg("--data")
    if pattern is None:
        print(__doc__)
        sys.exit(1)

    init_from = arg("--init")
    blocks = arg("--blocks", 12, int)
    channels = arg("--channels", 256, int)
    out_prefix = arg("--out", "checkpoints/lc0")
    lr = arg("--lr", 5e-4, float)
    value_mode = arg("--value", "blend")
    value_weight = arg("--value-weight", 1.0, float)
    batch = arg("--batch", 512, int)
    top_k = arg("--top", 30, int)
    max_steps = arg("--steps", 500_000, int)
    clip = arg("--clip", 1.0, float)

    if value_mode not in ("result", "best", "blend"):
        print("--value must be result, best or blend")
        sys.exit(1)

    latest_path = out_prefix + "_latest.pt"
    best_path = out_prefix + "_best.pt"
    os.makedirs(os.path.dirname(latest_path) or ".", exist_ok=True)

    use_amp = device_supports_amp() and "--fp32" not in sys.argv
    dtype = amp_dtype()
    device = pick_device()
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    print("mixed precision:",
          f"on ({str(dtype).split('.')[-1]})" if use_amp else "off")

    record_dtype = make_dtype(top_k)
    parts = load_records(pattern, record_dtype)

    # Hold out the tail of the last file, so validation never overlaps
    # training even when several files are mapped.
    n_val = min(VAL_POSITIONS, len(parts[-1]) // 10)
    val_data = parts[-1][-n_val:]
    parts[-1] = parts[-1][:-n_val]
    total = sum(len(p) for p in parts)
    print(f"\ntraining on {total:,}, validating on {len(val_data):,}")

    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        blocks, channels = ckpt.get("blocks", blocks), ckpt.get("channels",
                                                                channels)
        model = ChessNetV2(blocks, channels).to(device)
        model.load_state_dict(ckpt["model"])
        step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"resumed from {latest_path} at step {step:,}")
    elif init_from:
        ckpt = torch.load(init_from, map_location=device)
        if ckpt.get("encoding") != "v2":
            print(f"{init_from} is not a v2 checkpoint")
            sys.exit(1)
        blocks, channels = ckpt.get("blocks", blocks), ckpt.get("channels",
                                                                channels)
        model = ChessNetV2(blocks, channels).to(device)
        model.load_state_dict(ckpt["model"])
        step = 0
        best_loss = float("inf")
        print(f"initialised from {init_from} "
              f"(its step {ckpt.get('step', 0):,}), training restarts at 0")
        ckpt = {}
    else:
        ckpt = {}
        model = ChessNetV2(blocks, channels).to(device)
        step = 0
        best_loss = float("inf")
        print("starting from random weights")

    params = sum(p.numel() for p in model.parameters())
    print(f"model: {blocks} blocks x {channels} channels, "
          f"{params/1e6:.1f}M parameters")
    print(f"lr {lr}, value target '{value_mode}', value weight "
          f"{value_weight}, batch {batch}, grad clip {clip or 'off'}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    needs_scaler = use_amp and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

    steps_per_epoch = total // batch
    print(f"one pass over the data = {steps_per_epoch:,} steps\n")

    def save(path, with_opt):
        blob = {"model": model.state_dict(), "step": step,
                "blocks": blocks, "channels": channels,
                "best_loss": best_loss, "encoding": "v2",
                "trained_on": "lc0", "value_mode": value_mode}
        if with_opt:
            blob["opt"] = opt.state_dict()
            blob["scaler"] = scaler.state_dict()
        torch.save(blob, path)

    rng = np.random.default_rng()
    running_policy = running_value = 0.0
    skipped = 0
    last_log = time.time()

    try:
        while not max_steps or step < max_steps:
            recs = draw(parts, batch, rng)
            x, idx, prob, value = to_device(recs, device, value_mode)

            with torch.autocast("cuda", dtype=dtype, enabled=use_amp):
                logits, predicted = model(x)
                p_loss = policy_loss(logits, idx, prob)
                v_loss = F.mse_loss(predicted.float(), value)
                loss = p_loss + value_weight * v_loss

            if not torch.isfinite(loss):
                skipped += 1
                if skipped in (1, 10, 100) or skipped % 1000 == 0:
                    print(f"    skipped {skipped} non-finite batches")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()

            running_policy += p_loss.item()
            running_value += v_loss.item()
            step += 1

            if step % LOG_EVERY == 0:
                rate = LOG_EVERY / (time.time() - last_log)
                print(f"step {step:>8,} | policy {running_policy/LOG_EVERY:6.3f}"
                      f" | value {running_value/LOG_EVERY:7.5f} "
                      f"| {rate:5.1f} steps/s "
                      f"| {step/steps_per_epoch:5.2f} epochs")
                running_policy = running_value = 0.0
                last_log = time.time()

            if step % EVAL_EVERY == 0:
                p, v, top1 = assess(model, val_data, device, value_mode, batch)
                combined = p + value_weight * v
                marker = ""
                if combined < best_loss:
                    best_loss = combined
                    save(best_path, with_opt=False)
                    marker = "   <- new best, saved"
                print(f"    validation: policy {p:.4f}  value {v:.5f}  "
                      f"combined {combined:.4f}  top move {top1*100:.1f}%"
                      f"{marker}")
                last_log = time.time()

            if step % SAVE_EVERY == 0:
                save(latest_path, with_opt=True)

    except KeyboardInterrupt:
        print("\nstopping...")

    save(latest_path, with_opt=True)
    print(f"saved at step {step:,}  (best combined loss {best_loss:.4f})")


if __name__ == "__main__":
    main()
