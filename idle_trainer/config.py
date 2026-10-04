"""Persisted settings for the idle-time trainer menu bar app.

Two locations matter, and normally they're the same directory:

- PROJECT_ROOT: where train.py/play.py/search_batched.py/etc. actually
  live - needed on sys.path so this package can import/subprocess them.
- USER_DATA_DIR: where config.json, checkpoints/, and the dataset picker's
  default folder live - i.e. anything this app writes or that's specific
  to one person's data, as opposed to the app's own read-only code.

Running this straight from the project (the normal `-m idle_trainer` dev
setup) they're identical, computed exactly as before - nothing about that
path changes. The share-ready, self-contained .app sets
IDLE_TRAINER_SCRIPTS_DIR (its own bundled copy of the scripts, read-only)
and IDLE_TRAINER_DATA_DIR (a writable per-user folder under ~/Library/
Application Support, since a /Applications install has no business
writing into its own bundle) to pull them apart.
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("IDLE_TRAINER_SCRIPTS_DIR") or Path(__file__).resolve().parent.parent)

_installed_data_dir = os.environ.get("IDLE_TRAINER_DATA_DIR")
if _installed_data_dir:
    USER_DATA_DIR = Path(_installed_data_dir)
    CONFIG_PATH = USER_DATA_DIR / "config.json"
else:
    USER_DATA_DIR = PROJECT_ROOT
    CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# name -> (blocks, channels), matching train.py's ChessNet knobs.
NET_SIZE_PRESETS = {
    "Small": (4, 64),
    "Medium": (6, 128),
    "Large": (10, 192),
}

DEFAULT_IDLE_THRESHOLD_SECONDS = 180

# "auto"   - train only once the Mac has been idle past the threshold
# "always" - train continuously regardless of idle time
# "paused" - never train, until switched back
MODES = ["auto", "always", "paused"]

DEFAULT_MAX_GPU_PCT = 100      # target GPU duty cycle during training
DEFAULT_MAX_GPU_MEM_PCT = 100  # cap on the MPS allocator's memory ceiling

DEFAULTS = {
    "dataset": "data/positions.bin",
    "net_size": "Medium",
    "idle_threshold_seconds": DEFAULT_IDLE_THRESHOLD_SECONDS,
    "mode": "auto",
    "max_gpu_pct": DEFAULT_MAX_GPU_PCT,
    "max_gpu_mem_pct": DEFAULT_MAX_GPU_MEM_PCT,
}


def load_config():
    config = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            config.update(json.loads(CONFIG_PATH.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return config


def save_config(config):
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def is_compatible_dataset(path):
    """train.py always reads the file as an array of its fixed-size RECORD
    struct (see train.py) - a file built in any other layout (e.g. the
    eval-style records extract_evals.py/train_evals.py use, which end in a
    2-byte centipawn field instead of a 1-byte result) will very likely not
    divide evenly, and np.memmap raises immediately when it doesn't. That's
    a cheap, format-agnostic way to catch a wrong-format file before ever
    handing it to train.py, instead of after a crash."""
    try:
        from train import RECORD

        size = Path(path).stat().st_size
        return size > 0 and size % RECORD.itemsize == 0
    except OSError:
        return False


def ckpt_dir_for(config):
    """Each dataset/net-size combo gets its own checkpoint dir, so switching
    config in the app can never resume into (or overwrite) a checkpoint that
    was trained with different data or a different net shape."""
    stem = Path(config["dataset"]).stem
    size_key = config["net_size"].lower()
    return str(USER_DATA_DIR / "checkpoints" / "idle_trainer" / f"{stem}_{size_key}")
