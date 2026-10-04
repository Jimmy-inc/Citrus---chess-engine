"""
fetch_lc0.py - build a large Leela dataset without ever needing the space
for it.

Each archive is downloaded, converted, and deleted before the next one
starts, so peak disk use is about one archive plus its extracted chunks -
roughly a gigabyte - however many you end up processing. The converted
records are a third the size of the archive, because seven of the eight
history positions and all but the top moves are discarded.

It is resumable: an archive whose .bin already exists is skipped, so if the
job is killed you can simply start it again.

Usage:
    python fetch_lc0.py --budget 18
    python fetch_lc0.py --budget 18 --match run2
    python fetch_lc0.py --list

Options:
    --budget    gigabytes of converted output to stop at      (default 15)
    --match     only archives whose name contains this text
    --min-size  ignore archives smaller than this many MB     (default 100)
    --out       directory for the .bin files              (default data/lc0)
    --work      scratch directory for downloads          (default data/_work)
    --top       policy slots per record, must match training   (default 30)
    --list      show what would be fetched, then stop
    --keep-tar  do not delete archives after converting

Many of the archives listed are 10KB stubs rather than real data, which is
what --min-size filters out.
"""

import os
import re
import shutil
import subprocess
import sys
import time
INDEX = "https://storage.lczero.org/files/training_data/"
MB = 1024 * 1024
GB = 1024 * MB


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def listing():
    """
    Archive names and byte sizes from the directory index.

    Fetched with curl rather than urllib: the server rejects Python's
    default user agent with a 403.
    """
    result = subprocess.run(["curl", "-sfL", INDEX], capture_output=True,
                            timeout=120)
    if result.returncode != 0:
        print(f"could not fetch the index (curl exit {result.returncode})")
        sys.exit(1)
    html = result.stdout.decode("utf-8", "replace")

    found = []
    for line in html.splitlines():
        match = re.search(r'href="([^"]+\.tar)"', line)
        if not match:
            continue
        name = match.group(1)
        # The size is the last number on the line.
        sizes = re.findall(r"(\d+)\s*$", line.strip())
        size = int(sizes[-1]) if sizes else 0
        found.append((name, size))
    return found


def quota_used_mb():
    """Megabytes used against the home quota, or None if unavailable."""
    try:
        out = subprocess.run(["quota", "-s"], capture_output=True, text=True,
                             timeout=20).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, None
    for line in out.splitlines():
        numbers = re.findall(r"(\d+(?:\.\d+)?)([KMGT])", line)
        if len(numbers) >= 3:
            def to_mb(pair):
                value, unit = float(pair[0]), pair[1]
                return value * {"K": 1 / 1024, "M": 1, "G": 1024,
                                "T": 1024 * 1024}[unit]
            return to_mb(numbers[0]), to_mb(numbers[1])
    return None, None


def run(command):
    result = subprocess.run(command, shell=True)
    return result.returncode == 0


def main():
    budget_gb = arg("--budget", 15.0, float)
    match = arg("--match")
    min_size_mb = arg("--min-size", 100.0, float)
    out_dir = arg("--out", "data/lc0")
    work_dir = arg("--work", "data/_work")
    top_k = arg("--top", 30, int)
    keep_tar = "--keep-tar" in sys.argv

    archives = listing()
    real = [(n, s) for n, s in archives if s >= min_size_mb * MB]
    if match:
        real = [(n, s) for n, s in real if match in n]
    real.sort()

    print(f"{len(archives)} archives listed, {len(real)} above "
          f"{min_size_mb:.0f}MB" + (f" matching '{match}'" if match else ""))

    if "--list" in sys.argv:
        for name, size in real:
            print(f"  {size/MB:8.0f} MB  {name}")
        total = sum(s for _, s in real)
        print(f"\ntotal {total/GB:.1f} GB of archives, "
              f"about {total/GB/3:.1f} GB once converted")
        return

    os.makedirs(out_dir, exist_ok=True)
    used, limit = quota_used_mb()
    if used is not None:
        print(f"quota: {used/1024:.1f} GB used of {limit/1024:.1f} GB")
    print(f"budget: {budget_gb:.1f} GB of converted output\n")

    written = sum(os.path.getsize(os.path.join(out_dir, f))
                  for f in os.listdir(out_dir) if f.endswith(".bin"))
    if written:
        print(f"already have {written/GB:.2f} GB converted\n")

    done = failed = 0
    start = time.time()

    for name, size in real:
        if written >= budget_gb * GB:
            print(f"\nbudget reached ({written/GB:.2f} GB)")
            break

        stem = name[:-4]
        destination = os.path.join(out_dir, stem + ".bin")
        if os.path.exists(destination):
            print(f"skip   {stem}  (already converted)")
            continue

        # Leave room for the archive plus its extracted contents.
        if limit is not None:
            used, limit = quota_used_mb()
            headroom = (limit - used) * MB
            if headroom < size * 2.5:
                print(f"\nstopping: only {headroom/GB:.1f} GB of quota free, "
                      f"need about {size*2.5/GB:.1f} GB to process {stem}")
                break

        print(f"\n--- {stem}  ({size/MB:.0f} MB) ---")
        shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
        archive = os.path.join(work_dir, name)

        if not run(f'curl -sf -o "{archive}" "{INDEX}{name}"'):
            print("  download failed")
            failed += 1
            shutil.rmtree(work_dir, ignore_errors=True)
            continue

        chunks = os.path.join(work_dir, "chunks")
        os.makedirs(chunks, exist_ok=True)
        if not run(f'tar -xf "{archive}" -C "{chunks}"'):
            print("  extract failed")
            failed += 1
            shutil.rmtree(work_dir, ignore_errors=True)
            continue

        if not keep_tar:
            os.remove(archive)

        partial = destination + ".partial"
        ok = run(f'./.venv/bin/python convert_lc0.py "{chunks}" "{partial}" '
                 f'--top {top_k}')
        if ok and os.path.exists(partial) and os.path.getsize(partial) > 0:
            # Only take the final name once conversion finished, so an
            # interrupted run never leaves a truncated file that a later
            # run would skip over.
            os.rename(partial, destination)
            written += os.path.getsize(destination)
            done += 1
        else:
            print("  conversion failed")
            failed += 1
            if os.path.exists(partial):
                os.remove(partial)

        shutil.rmtree(work_dir, ignore_errors=True)

        elapsed = (time.time() - start) / 60
        print(f"  running total: {written/GB:.2f} GB from {done} archives, "
              f"{elapsed:.0f} min elapsed")

    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"\n{'=' * 56}")
    print(f"converted {done} archives, {failed} failed")
    print(f"{written/GB:.2f} GB in {out_dir}")
    print(f"took {(time.time()-start)/60:.0f} min")
    print(f"\ntrain on all of it with:")
    print(f"  python train_lc0.py --data '{out_dir}/*.bin' "
          f"--init checkpoints/v2_best.pt")


if __name__ == "__main__":
    main()
