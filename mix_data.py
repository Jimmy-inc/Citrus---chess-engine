"""
mix_data.py - blend the weakness records with ordinary training data, so a
fine-tune learns from the net's mistakes without forgetting everything else.

Usage:
    python mix_data.py --weak data/weakness_all.bin --out data/mix25.bin
    python mix_data.py --weak data/weakness_all.bin --ratio 0.5 --out data/mix50.bin
    python mix_data.py --weak 'data/weak2_*.bin' --out data/mix.bin

Training only on positions the net got wrong pulls it towards that
distribution with nothing holding its evaluations elsewhere in place, and it
drifts - the way the Leela fine-tune came out weaker. Mixing in a sample of
what the net was originally trained on (rehearsal) is the standard guard.

The ordinary records come from all over --base, in small blocks rather than
one at a time. Single records would touch nearly every page of a 14GB file
on a network drive; blocks of 64 read a small fraction of it and still come
from tens of thousands of places.

The output is shuffled, which matters beyond training order: train_v2.py
holds out the *last* records as validation, and after shuffling that tail is
a fair sample of the mix rather than whichever input happened to come last.

Weakness records are deduplicated on the way in. Careful with globs here:
weakness_all.bin already contains weakness.bin and weakness2.bin, so
'data/weakness*.bin' would read everything twice. The script reports how
many duplicates it dropped and stops if it looks like inputs overlap.

Options:
    --weak     weakness records to include; quote globs         (required)
    --out      where to write the mix                           (required)
    --base     ordinary records to sample from
                                             (default data/evals_d22.bin)
    --ratio    fraction of the output that is weakness data   (default 0.25)
    --block    consecutive base records taken per draw          (default 64)
    --seed     random seed, so the same mix can be rebuilt       (default 0)
    --force    overwrite --out, and carry on despite overlapping inputs
"""

import glob
import os
import sys
import time

import numpy as np

# Must match EVAL_RECORD in extract_evals.py, and so train_v2.py.
EVAL_RECORD = np.dtype([
    ("pieces",   np.int8, (64,)),
    ("stm",      np.int8),
    ("castling", np.int8),
    ("ep",       np.int8),
    ("from_sq",  np.int8),
    ("to_sq",    np.int8),
    ("promo",    np.int8),
    ("cp",       np.int16),   # centipawns, from the side to move's view
])

BATCH = 512          # train_v2.py's default, for the steps-per-epoch figure
OVERLAP_WARNING = 0.10


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def records(path):
    """Memory-map a record file, refusing anything that is not whole records."""
    size = os.path.getsize(path)
    if size % EVAL_RECORD.itemsize:
        print(f"{path} is {size:,} bytes, not a whole number of "
              f"{EVAL_RECORD.itemsize}-byte records - wrong format?")
        sys.exit(1)
    return np.memmap(path, dtype=EVAL_RECORD, mode="r")


def unique(recs):
    """Drop exact duplicate records, keeping first sightings in order."""
    as_bytes = recs.view(np.dtype((np.void, EVAL_RECORD.itemsize)))
    _, first = np.unique(as_bytes, return_index=True)
    return recs[np.sort(first)]


def sample_blocks(base, count, block, rng):
    """`count` records from `base`, taken as whole non-overlapping blocks.

    Blocks are chosen from a fixed grid without replacement, so no record
    can be drawn twice.
    """
    slots = len(base) // block
    needed = -(-count // block)
    if needed > slots:
        print(f"asked for {count:,} base records but --base only holds "
              f"{slots * block:,} in whole blocks of {block}")
        sys.exit(1)
    chosen = np.sort(rng.choice(slots, size=needed, replace=False))
    picked = np.concatenate([base[s * block:(s + 1) * block] for s in chosen])
    return picked[:count]


def main():
    pattern = arg("--weak")
    out_path = arg("--out")
    if not pattern or not out_path:
        print(__doc__)
        sys.exit(1)

    base_path = arg("--base", "data/evals_d22.bin")
    ratio = arg("--ratio", 0.25, float)
    block = arg("--block", 64, int)
    seed = arg("--seed", 0, int)
    force = "--force" in sys.argv

    if not 0 < ratio <= 1:
        print("--ratio is the weakness share of the output, between 0 and 1")
        sys.exit(1)

    # Checked before any reading, so a clash costs nothing.
    if os.path.exists(out_path) and not force:
        print(f"{out_path} already exists. Pick another name, or --force.")
        sys.exit(1)

    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no files match {pattern!r} (quote the glob or the shell "
              f"eats it)")
        sys.exit(1)
    if os.path.abspath(out_path) in {os.path.abspath(p) for p in paths}:
        print(f"{out_path} is also one of the inputs")
        sys.exit(1)

    print(f"weakness from {len(paths)} file(s) matching {pattern!r}:")
    for p in paths:
        print(f"  {p}  {os.path.getsize(p) // EVAL_RECORD.itemsize:>10,} records")
    print(f"ordinary from {base_path}  |  {ratio:.0%} weakness  |  "
          f"blocks of {block}  |  seed {seed}")
    print(f"writing {out_path}\n")

    start = time.time()
    weak = np.concatenate([np.array(records(p)) for p in paths])
    read = len(weak)
    weak = unique(weak)
    dropped = read - len(weak)
    print(f"weakness records : {read:,} read, {dropped:,} duplicates dropped")

    if read and dropped / read > OVERLAP_WARNING and not force:
        print(f"\n{dropped / read:.0%} of the weakness records were "
              f"duplicates. That usually means the inputs overlap - for "
              f"instance weakness_all.bin alongside the files it was built "
              f"from. Check --weak, or pass --force if this is intended.")
        sys.exit(1)

    wanted = round(len(weak) * (1 - ratio) / ratio)
    base = records(base_path)
    rng = np.random.default_rng(seed)
    ordinary = (sample_blocks(base, wanted, block, rng) if wanted
                else np.zeros(0, dtype=EVAL_RECORD))
    print(f"ordinary records : {len(ordinary):,} sampled from "
          f"{len(base):,} in {time.time() - start:.0f}s")

    mix = np.concatenate([weak, ordinary])
    mix = mix[rng.permutation(len(mix))]
    mix.tofile(out_path)

    total = len(mix)
    epoch = -(-total // BATCH)
    print(f"\nDone in {time.time() - start:.0f}s")
    print(f"  weakness   : {len(weak):>10,}  ({len(weak) / total:.1%})")
    print(f"  ordinary   : {len(ordinary):>10,}  ({len(ordinary) / total:.1%})")
    print(f"  total      : {total:>10,}")
    print(f"  file size  : {total * EVAL_RECORD.itemsize / 1e6:.1f} MB")
    print(f"  one epoch  : {epoch:,} steps at batch {BATCH}")

    print("\nBefore training, seed a new checkpoint with its step count reset"
          " -\ntrain_v2.py's --steps is an absolute target, so a resumed "
          "checkpoint\nalready past it trains nothing.")


if __name__ == "__main__":
    main()
