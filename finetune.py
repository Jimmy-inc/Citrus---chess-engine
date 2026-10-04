"""
finetune.py - continue training an existing net on a different dataset,
at a low learning rate, without touching the original checkpoints.

Usage:
    python finetune.py --data data/tcec.bin
    python finetune.py --data data/tcec.bin --lr 5e-5 --steps 30000
    python finetune.py --data data/tcec.bin --sleep 0.08

Options:
    --data   dataset to fine-tune on          (required)
    --init   weights to start from            (default checkpoints/best.pt)
    --out    where to write                   (default checkpoints/finetuned)
    --lr     learning rate                    (default 1e-4)
    --steps  stop after this many steps       (default 20000, 0 = never)
    --sleep  idle seconds per step

Writes <out>_latest.pt and <out>_best.pt. Your original checkpoints are
left alone, so you can play the two nets against each other afterwards.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from train import (ChessNet, pick_device, RECORD, BATCH, VALUE_WEIGHT, VAL_POSITIONS,
                   LOG_EVERY, EVAL_EVERY, SAVE_EVERY,
                   batch_to_device, evaluate)


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def main():
    data_path = arg("--data")
    if data_path is None:
        print(__doc__)
        sys.exit(1)

    init_path = arg("--init", "checkpoints/best.pt")
    out_prefix = arg("--out", "checkpoints/finetuned")
    lr = arg("--lr", 1e-4, float)
    max_steps = arg("--steps", 20_000, int)
    sleep_per_step = arg("--sleep", 0.0, float)

    latest_path = out_prefix + "_latest.pt"
    best_path = out_prefix + "_best.pt"

    device = pick_device()
    print("device:", device)

    data = np.memmap(data_path, dtype=RECORD, mode="r")
    n_val = min(VAL_POSITIONS, len(data) // 100)
    train_data = data[:-n_val]
    val_data = data[-n_val:]
    print(f"fine-tuning on {data_path}")
    print(f"train positions: {len(train_data):,}   validation: {len(val_data):,}")

    # Start from the existing weights. The old optimizer state is deliberately
    # discarded: its momentum is tuned to the old data distribution.
    if os.path.exists(latest_path):
        source, resuming = latest_path, True
    else:
        source, resuming = init_path, False

    if not os.path.exists(source):
        print(f"no checkpoint at {source}")
        sys.exit(1)

    ckpt = torch.load(source, map_location=device)
    model = ChessNet(blocks=ckpt.get("blocks", 6),
                     channels=ckpt.get("channels", 128)).to(device)
    model.load_state_dict(ckpt["model"])

    step = ckpt.get("finetune_step", 0) if resuming else 0
    best_vloss = ckpt.get("best_vloss", float("inf")) if resuming else float("inf")

    print(f"{'resumed' if resuming else 'initialised'} from {source}"
          f" (base step {ckpt.get('step', 0):,})")
    print(f"learning rate {lr}, stopping at "
          f"{max_steps:,} steps" if max_steps else "running until stopped")
    if sleep_per_step:
        print(f"throttled: idling {sleep_per_step*1000:.0f} ms after each step")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if resuming and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])

    # Establish the baseline before any fine-tuning happens.
    vloss, acc = evaluate(model, val_data, device)
    print(f"\nbefore fine-tuning: loss {vloss:.4f}  move match {acc*100:.1f}%\n")

    def save(path, with_opt):
        blob = {"model": model.state_dict(),
                "step": ckpt.get("step", 0),
                "finetune_step": step,
                "blocks": ckpt.get("blocks", 6),
                "channels": ckpt.get("channels", 128),
                "best_vloss": best_vloss}
        if with_opt:
            blob["opt"] = opt.state_dict()
        torch.save(blob, path)

    rng = np.random.default_rng()
    running = 0.0
    last_log = time.time()

    try:
        while not max_steps or step < max_steps:
            idx = np.sort(rng.integers(0, len(train_data), BATCH))
            x, move, value = batch_to_device(train_data[idx], device)

            p, v = model(x)
            loss = F.cross_entropy(p, move) + VALUE_WEIGHT * F.mse_loss(v, value)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item()
            step += 1

            if sleep_per_step:
                time.sleep(sleep_per_step)

            if step % LOG_EVERY == 0:
                rate = LOG_EVERY / (time.time() - last_log)
                print(f"step {step:>7,} | loss {running/LOG_EVERY:6.3f} | "
                      f"{rate:5.1f} steps/s")
                running = 0.0
                last_log = time.time()

            if step % EVAL_EVERY == 0:
                vloss, acc = evaluate(model, val_data, device)
                marker = ""
                if vloss < best_vloss:
                    best_vloss = vloss
                    save(best_path, with_opt=False)
                    marker = "   <- new best, saved"
                print(f"    validation: loss {vloss:.4f}  "
                      f"move match {acc*100:.1f}%{marker}")
                last_log = time.time()

            if step % SAVE_EVERY == 0:
                save(latest_path, with_opt=True)

    except KeyboardInterrupt:
        print("\nstopping...")

    save(latest_path, with_opt=True)
    print(f"saved at fine-tune step {step:,}  (best {best_vloss:.4f})")
    print(f"  {latest_path}\n  {best_path}")


if __name__ == "__main__":
    main()
