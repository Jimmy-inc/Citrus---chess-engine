"""
train_v2.py - train the version two network on Stockfish evaluation records.

What is different from train_evals.py:

  * the board is canonicalized to the mover's point of view, so the net sees
    one problem instead of two mirrored ones;
  * the policy head is convolutional - 73 move-type planes over 8x8 - rather
    than a dense layer over all 4096 from/to pairs.

The second change is where the parameters move. In the old 6x128 net, 8.4M
of 10.3M parameters sat in the policy output layer and 1.8M in the trunk
that did the actual thinking. Here a 12x256 trunk holds about 14M and the
policy head about 0.6M, so nearly all the capacity is somewhere useful.

Usage:
    python train_v2.py --data data/evals_d22.bin
    python train_v2.py --data data/evals_d22.bin --blocks 12 --channels 256
    python train_v2.py --data data/evals_d22.bin --steps 0 --sleep 0.05

Options:
    --data          eval records from extract_evals.py        (required)
    --blocks        residual blocks                           (default 12)
    --channels      trunk width                              (default 256)
    --out           checkpoint prefix              (default checkpoints/v2)
    --lr            learning rate                          (default 1e-3)
    --value-weight  value loss weight against policy         (default 1.0)
    --scale         centipawns per unit of tanh input        (default 400)
    --batch         positions per step                       (default 512)
    --steps         stop after this many steps, 0 = never  (default 200000)
    --sleep         idle seconds per step, to run cooler       (default 0)
    --fp32          disable mixed precision (CUDA only)
    --clip          gradient norm clip, 0 disables            (default 1.0)

bfloat16 is preferred over float16 where the card supports it. It has the
same exponent range as float32, so activations cannot overflow the way they
can in float16 - which is what silently turns a network to NaN partway
through a long run.

Mixed precision is on by default on CUDA. The tensor cores on an A4000 sit
idle in fp32, and turning them on is worth roughly two to three times the
throughput on a convolutional net this size, for no measurable loss in
quality. Gradients are scaled to stop small values underflowing in fp16.
"""

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding_v2 import (PLANES, POLICY_SIZE, POLICY_PLANES, MOVE_LOOKUP,
                         canonicalize, planes_from_pieces)

VAL_POSITIONS = 200_000
LOG_EVERY = 100
EVAL_EVERY = 2_000
SAVE_EVERY = 2_000

# Must match EVAL_RECORD in extract_evals.py.
EVAL_RECORD = np.dtype([
    ("pieces",   np.int8, (64,)),
    ("stm",      np.int8),
    ("castling", np.int8),
    ("ep",       np.int8),
    ("from_sq",  np.int8),
    ("to_sq",    np.int8),
    ("promo",    np.int8),
    ("cp",       np.int16),
])


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_supports_amp():
    return torch.cuda.is_available()


def amp_dtype():
    """
    bfloat16 when the card has it, float16 otherwise.

    float16 has only 5 exponent bits, so a large activation overflows to
    infinity and poisons every weight downstream. bfloat16 keeps float32's
    range at the cost of precision that a training loop does not need.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


# ---------------------------------------------------------------- model


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class ChessNetV2(nn.Module):
    def __init__(self, blocks=12, channels=256):
        super().__init__()
        self.blocks_count = blocks
        self.channels = channels

        self.stem = nn.Sequential(
            nn.Conv2d(PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(blocks)])

        # Policy stays convolutional right to the output: one channel per
        # move type, one square per origin. A knight move learned on d4 is
        # the same weights as the one on e5.
        self.policy = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, POLICY_PLANES, 1),
        )

        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(8 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.blocks(self.stem(x))

        policy = self.policy(x)                       # (B, 73, 8, 8)
        # Rearrange so the flat index reads square * 73 + plane, matching
        # the lookup table in encoding_v2.
        policy = policy.permute(0, 2, 3, 1).reshape(-1, POLICY_SIZE)

        value = self.value_fc(self.value_conv(x).flatten(1)).squeeze(1)
        return policy, value


# ---------------------------------------------------------------- data


def castle_onto_rook(pieces, from_sq, to_sq):
    """
    Castling written as e1g1 rewritten as e1h1 (and e1c1 as e1a1).

    Lichess's database, which most records come from, writes castling with
    the king landing on its rook, so that is where the nets learned it.
    Records written by python-chess (self-play, label_weak.py) use e1g1.
    One spelling for all of them, so they teach the same output. Works on
    canonical boards, where the mover's king is 6 and starts on e1.
    """
    rows = np.arange(len(pieces))
    king = pieces[rows, from_sq.astype(np.int64)] == 6
    castle = king & (from_sq == 4) & ((to_sq == 6) | (to_sq == 2))
    return np.where(castle, np.where(to_sq == 6, 7, 0), to_sq)


def expand(recs, scale):
    """Canonical planes, policy targets and value targets for a batch."""
    pieces, castling, ep, from_sq, to_sq = canonicalize(
        recs["pieces"], recs["stm"], recs["castling"], recs["ep"],
        recs["from_sq"], recs["to_sq"])
    to_sq = castle_onto_rook(pieces, from_sq, to_sq)

    x = planes_from_pieces(pieces, castling, ep)

    promo = recs["promo"].astype(np.int64)
    target = MOVE_LOOKUP[from_sq.astype(np.int64),
                         to_sq.astype(np.int64),
                         promo].astype(np.int64)

    # A move the geometry cannot express should not happen for legal moves,
    # but if one slips through, park it on index 0 and mask it out of the
    # loss rather than letting it train on nonsense.
    valid = target >= 0
    target = np.where(valid, target, 0)

    value = np.tanh(recs["cp"].astype(np.float32) / scale)

    return x, target, value, valid


def to_device(recs, device, scale):
    x, target, value, valid = expand(recs, scale)
    return (torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(target).to(device, non_blocking=True),
            torch.from_numpy(value).to(device, non_blocking=True),
            torch.from_numpy(valid).to(device, non_blocking=True))


@torch.no_grad()
def assess(model, val, device, scale, batch, batches=20):
    model.eval()
    correct = seen = 0
    error = 0.0
    for i in range(batches):
        recs = val[i * batch:(i + 1) * batch]
        if len(recs) == 0:
            break
        x, target, value, valid = to_device(recs, device, scale)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=torch.cuda.is_available()):
            policy, predicted = model(x)
        predicted = predicted.float()
        correct += ((policy.argmax(1) == target) & valid).sum().item()
        error += (predicted - value).abs().sum().item()
        seen += len(recs)
    model.train()
    return correct / seen, error / seen


# ---------------------------------------------------------------- main


def main():
    data_path = arg("--data")
    if data_path is None:
        print(__doc__)
        sys.exit(1)

    blocks = arg("--blocks", 12, int)
    channels = arg("--channels", 256, int)
    out_prefix = arg("--out", "checkpoints/v2")
    lr = arg("--lr", 1e-3, float)
    value_weight = arg("--value-weight", 1.0, float)
    scale = arg("--scale", 400.0, float)
    batch = arg("--batch", 512, int)
    max_steps = arg("--steps", 200_000, int)
    sleep_per_step = arg("--sleep", 0.0, float)
    clip = arg("--clip", 1.0, float)

    latest_path = out_prefix + "_latest.pt"
    best_path = out_prefix + "_best.pt"
    os.makedirs(os.path.dirname(latest_path) or ".", exist_ok=True)

    use_amp = device_supports_amp() and "--fp32" not in sys.argv

    device = pick_device()
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        # Input shapes never change here, so let cuDNN pick the fastest
        # algorithm once rather than re-deciding every step.
        torch.backends.cudnn.benchmark = True
    dtype = amp_dtype()
    print("mixed precision:",
          f"on ({str(dtype).split('.')[-1]})" if use_amp else "off")

    data = np.memmap(data_path, dtype=EVAL_RECORD, mode="r")
    n_val = min(VAL_POSITIONS, len(data) // 100)
    train_data = data[:-n_val]
    val_data = data[-n_val:]
    print(f"train positions: {len(train_data):,}   validation: {len(val_data):,}")

    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        blocks = ckpt.get("blocks", blocks)
        channels = ckpt.get("channels", channels)
        model = ChessNetV2(blocks, channels).to(device)
        model.load_state_dict(ckpt["model"])
        step = ckpt.get("step", 0)
        best_error = ckpt.get("best_error", float("inf"))
        print(f"resumed from {latest_path} at step {step:,}")
    else:
        ckpt = {}
        model = ChessNetV2(blocks, channels).to(device)
        step = 0
        best_error = float("inf")
        print("starting from random weights")

    params = sum(p.numel() for p in model.parameters())
    trunk = sum(p.numel() for p in model.blocks.parameters())
    policy_params = sum(p.numel() for p in model.policy.parameters())
    print(f"model: {blocks} blocks x {channels} channels, "
          f"{params/1e6:.1f}M parameters")
    print(f"  trunk {trunk/1e6:.1f}M, policy head {policy_params/1e6:.1f}M")
    print(f"lr {lr}, value weight {value_weight}, scale {scale}cp, "
          f"batch {batch}, grad clip {clip or 'off'}")
    if sleep_per_step:
        print(f"throttled: idling {sleep_per_step*1000:.0f} ms per step")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])

    # A scaler is only needed for float16; bfloat16 does not underflow the
    # way float16 does, so scaling would be pointless work.
    needs_scaler = use_amp and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)
    if "scaler" in ckpt and use_amp:
        scaler.load_state_dict(ckpt["scaler"])

    steps_per_epoch = len(train_data) // batch
    print(f"one pass over the data = {steps_per_epoch:,} steps\n")

    def save(path, with_opt):
        blob = {"model": model.state_dict(), "step": step,
                "blocks": blocks, "channels": channels,
                "best_error": best_error,
                "encoding": "v2", "scale": scale}
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
            idx = np.sort(rng.integers(0, len(train_data), batch))
            x, target, value, valid = to_device(train_data[idx], device, scale)

            if not bool(valid.any()):
                continue          # nothing to learn from; skip the batch

            with torch.autocast("cuda", dtype=dtype, enabled=use_amp):
                policy, predicted = model(x)
                policy_loss = F.cross_entropy(policy[valid], target[valid])
                value_loss = F.mse_loss(predicted.float(), value)
                loss = policy_loss + value_weight * value_loss

            if not torch.isfinite(loss):
                # One bad batch should cost a step, not the whole run.
                skipped += 1
                if skipped in (1, 10, 100) or skipped % 1000 == 0:
                    print(f"    skipped {skipped} non-finite batches so far")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if clip:
                # Unscale first, or the clip threshold would be applied to
                # scaled gradients and mean nothing.
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            scaler.step(opt)
            scaler.update()

            running_policy += policy_loss.item()
            running_value += value_loss.item()
            step += 1

            if sleep_per_step:
                time.sleep(sleep_per_step)

            if step % LOG_EVERY == 0:
                rate = LOG_EVERY / (time.time() - last_log)
                epochs = step / steps_per_epoch
                print(f"step {step:>8,} | policy {running_policy/LOG_EVERY:6.3f} "
                      f"| value {running_value/LOG_EVERY:7.5f} "
                      f"| {rate:5.1f} steps/s | {epochs:5.2f} epochs")
                running_policy = running_value = 0.0
                last_log = time.time()

            if step % EVAL_EVERY == 0:
                match, error = assess(model, val_data, device, scale, batch)
                marker = ""
                if error < best_error:
                    best_error = error
                    save(best_path, with_opt=False)
                    marker = "   <- new best, saved"
                print(f"    validation: policy {match*100:.1f}%  "
                      f"value error {error:.4f}{marker}")
                last_log = time.time()

            if step % SAVE_EVERY == 0:
                save(latest_path, with_opt=True)

    except KeyboardInterrupt:
        print("\nstopping...")

    save(latest_path, with_opt=True)
    print(f"saved at step {step:,}  (best value error {best_error:.4f})")


if __name__ == "__main__":
    main()
