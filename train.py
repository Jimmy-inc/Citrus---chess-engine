"""
train.py - train a policy/value network on the records made by extract.py.

Usage:
    python train.py                    # start or resume training
    python train.py --fresh            # ignore any existing checkpoint
    python train.py --sleep 0.08       # idle 80ms between steps to run cooler

Stop it any time with Control-C; it saves before exiting and picks up
where it left off next time you run it.

Two checkpoints are kept:
    checkpoints/latest.pt   most recent state, used to resume training
    checkpoints/best.pt     the weights with the lowest validation loss so far
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def pick_device():
    """CUDA if there is a card, then Apple Metal, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- settings

DATA_PATH = "data/positions.bin"
CKPT_DIR = "checkpoints"
CKPT_PATH = os.path.join(CKPT_DIR, "latest.pt")
BEST_PATH = os.path.join(CKPT_DIR, "best.pt")

BLOCKS = 6            # residual blocks
CHANNELS = 128        # width of the network
BATCH = 512
LR = 1e-3
VALUE_WEIGHT = 0.5    # how much the value head matters vs the policy head

SLEEP_PER_STEP = 0.0  # seconds idle after each step; raise to cut heat/noise

VAL_POSITIONS = 100_000   # held out from the end of the file
LOG_EVERY = 100
EVAL_EVERY = 2_000
SAVE_EVERY = 2_000

# ---------------------------------------------------------------- data

# Must match RECORD in extract.py exactly.
RECORD = np.dtype([
    ("pieces",   np.int8, (64,)),
    ("stm",      np.int8),
    ("castling", np.int8),
    ("ep",       np.int8),
    ("from_sq",  np.int8),
    ("to_sq",    np.int8),
    ("promo",    np.int8),
    ("result",   np.int8),
])

# 18 input planes: 12 piece types, 4 castling rights, side to move, en passant
PLANES = 18
EYE13 = np.eye(13, dtype=np.float32)


def expand(recs):
    """Turn a batch of 71-byte records into network inputs and targets."""
    b = len(recs)

    pieces = recs["pieces"].astype(np.int64)          # (B, 64)
    planes = EYE13[pieces][:, :, 1:]                  # (B, 64, 12), drop 'empty'
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

    # Policy target: 4096 classes, one per (from, to) pair.
    move = recs["from_sq"].astype(np.int64) * 64 + recs["to_sq"].astype(np.int64)

    # Value target: result from the perspective of whoever is to move.
    sign = np.where(recs["stm"] == 1, 1.0, -1.0).astype(np.float32)
    value = recs["result"].astype(np.float32) * sign

    return x, move, value


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


class ChessNet(nn.Module):
    def __init__(self, blocks=BLOCKS, channels=CHANNELS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(32 * 64, 4096)

        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(8 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.blocks(self.stem(x))
        p = self.policy_fc(self.policy_conv(x).flatten(1))
        v = self.value_fc(self.value_conv(x).flatten(1)).squeeze(1)
        return p, v


# ---------------------------------------------------------------- training


def batch_to_device(recs, device):
    x, move, value = expand(recs)
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(move).to(device),
        torch.from_numpy(value).to(device),
    )


@torch.no_grad()
def evaluate(model, val, device, batches=20):
    model.eval()
    total_loss = correct = seen = 0
    for i in range(batches):
        recs = val[i * BATCH:(i + 1) * BATCH]
        if len(recs) == 0:
            break
        x, move, value = batch_to_device(recs, device)
        p, v = model(x)
        loss = F.cross_entropy(p, move) + VALUE_WEIGHT * F.mse_loss(v, value)
        total_loss += loss.item() * len(recs)
        correct += (p.argmax(1) == move).sum().item()
        seen += len(recs)
    model.train()
    return total_loss / seen, correct / seen


def main():
    fresh = "--fresh" in sys.argv

    sleep_per_step = SLEEP_PER_STEP
    if "--sleep" in sys.argv:
        sleep_per_step = float(sys.argv[sys.argv.index("--sleep") + 1])

    device = pick_device()
    print("device:", device)

    data = np.memmap(DATA_PATH, dtype=RECORD, mode="r")
    n_val = min(VAL_POSITIONS, len(data) // 100)
    train_data = data[:-n_val]
    val_data = data[-n_val:]
    print(f"train positions: {len(train_data):,}   validation: {len(val_data):,}")

    model = ChessNet().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"model: {BLOCKS} blocks x {CHANNELS} channels, {params/1e6:.1f}M parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    step = 0
    best_vloss = float("inf")

    os.makedirs(CKPT_DIR, exist_ok=True)
    if os.path.exists(CKPT_PATH) and not fresh:
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        best_vloss = ckpt.get("best_vloss", float("inf"))
        print(f"resumed from step {step:,}")
        if best_vloss < float("inf"):
            print(f"best validation loss so far: {best_vloss:.4f}")

    steps_per_epoch = len(train_data) // BATCH
    print(f"one pass over the data = {steps_per_epoch:,} steps")
    if sleep_per_step:
        print(f"throttled: idling {sleep_per_step*1000:.0f} ms after each step")
    print()

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "blocks": BLOCKS, "channels": CHANNELS,
                    "best_vloss": best_vloss},
                   CKPT_PATH)

    def save_best():
        # Weights only - this copy is for playing, not for resuming training,
        # so there is no need to carry the optimizer state.
        torch.save({"model": model.state_dict(), "step": step,
                    "blocks": BLOCKS, "channels": CHANNELS,
                    "val_loss": best_vloss},
                   BEST_PATH)

    rng = np.random.default_rng()
    running = 0.0
    last_log = time.time()

    try:
        while True:
            idx = np.sort(rng.integers(0, len(train_data), BATCH))
            recs = train_data[idx]
            x, move, value = batch_to_device(recs, device)

            p, v = model(x)
            loss = F.cross_entropy(p, move) + VALUE_WEIGHT * F.mse_loss(v, value)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item()   # this also waits for the GPU to finish
            step += 1

            # Idle time here is real idle time for the GPU, which is what
            # brings the temperature down.
            if sleep_per_step:
                time.sleep(sleep_per_step)

            if step % LOG_EVERY == 0:
                elapsed = time.time() - last_log
                rate = LOG_EVERY / elapsed
                epochs = step / steps_per_epoch
                print(f"step {step:>8,} | loss {running/LOG_EVERY:6.3f} | "
                      f"{rate:5.1f} steps/s | {epochs:5.2f} epochs")
                running = 0.0
                last_log = time.time()

            if step % EVAL_EVERY == 0:
                vloss, acc = evaluate(model, val_data, device)
                marker = ""
                if vloss < best_vloss:
                    best_vloss = vloss
                    save_best()
                    marker = "   <- new best, saved"
                print(f"    validation: loss {vloss:.4f}  "
                      f"move match {acc*100:.1f}%{marker}")
                last_log = time.time()

            if step % SAVE_EVERY == 0:
                save()

    except KeyboardInterrupt:
        print("\nstopping, saving checkpoint...")
        save()
        print(f"saved at step {step:,}  (best validation loss {best_vloss:.4f})")


if __name__ == "__main__":
    main()
