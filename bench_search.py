"""
bench_search.py - how fast the search really runs on this machine, measured
the way the bot uses it: replaying real games and searching every second
move, so the network cache carries over between moves as it does in play.

Usage:
    python bench_search.py                      compare the code in the
                                                current folder with the
                                                code beside this script
    python bench_search.py --old ~/chessbot --new ~/chessbot_next
    python bench_search.py --sims 4000 --games 3 --pgn 'data/sp2_*.pgn'

Run it from the project folder, so checkpoints/ and data/ resolve. Each
variant runs in its own process, since two versions of search_v2 cannot be
imported side by side. The draw rule differs between old and new code, so
an occasional different move choice between them is expected, not a bug.

Options:
    --old     folder holding the code to compare against    (default .)
    --new     folder holding the new code  (default this script's folder)
    --pgn     games to replay; quote globs     (default data/sp2_*.pgn)
    --games   how many games                                  (default 2)
    --sims    simulations per search                       (default 3000)
    --ckpt    checkpoint                (default checkpoints/v2_best.pt)
"""

import glob
import json
import os
import subprocess
import sys
import time

# (label, code folder key, settings applied to the new code)
VARIANTS = [
    ("current code", "old", {}),
    ("new code", "new", {}),
    ("new, CPU priors", "new", {"PRIORS_ON_DEVICE": False}),
    ("new, 1 torch thread", "new", {"threads": 1}),
    ("new, no cache", "new", {"CACHE_SIZE": 0}),
]


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def worker():
    """One variant: replay the games and report speed as one JSON line."""
    code = arg("--code")
    settings = json.loads(arg("--settings", "{}"))
    sys.path.insert(0, os.path.abspath(code))
    # Modules the new code didn't change still live in the project folder.
    sys.path.append(os.getcwd())

    import torch
    import chess.pgn
    import search_v2 as E

    if "threads" in settings:
        torch.set_num_threads(settings.pop("threads"))
    for name, value in settings.items():
        setattr(E, name, value)

    model, _ = E.load(arg("--ckpt"))
    device = E.device_for(model)

    # Time spent inside the network, synchronised so it is real GPU time.
    spent = [0.0]
    forward = model.forward

    def timed(x):
        start = time.perf_counter()
        out = forward(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        spent[0] += time.perf_counter() - start
        return out
    model.forward = timed

    games = []
    for path in sorted(glob.glob(arg("--pgn"))):
        with open(path) as handle:
            while len(games) < arg("--games", cast=int):
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                if len(list(game.mainline_moves())) >= 50:
                    games.append(game)

    E.run_search(model, games[0].board(), 200)         # wake the GPU up
    spent[0] = 0.0
    sims = arg("--sims", cast=int)
    total = searches = 0.0
    choices = []
    for game in games:
        getattr(E, "clear_cache", lambda: None)()      # a fresh game
        board = game.board()
        for ply, move in enumerate(list(game.mainline_moves())[:50]):
            if ply % 2 == 0 and ply >= 10:
                start = time.perf_counter()
                root = E.run_search(model, board, sims)
                total += time.perf_counter() - start
                searches += 1
                choices.append(E.choose(root)[0].uci())
            board.push(move)

    done = sims * searches
    print(json.dumps({
        "rate": done / total,
        "network": spent[0] / total,
        "cpu_us": (total - spent[0]) / done * 1e6,
        "choices": choices,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "-",
        "threads": torch.get_num_threads(),
        "cores": (len(os.sched_getaffinity(0))
                  if hasattr(os, "sched_getaffinity") else os.cpu_count()),
    }))


def main():
    if "--code" in sys.argv:
        worker()
        return

    folders = {"old": arg("--old", "."),
               "new": arg("--new", os.path.dirname(os.path.abspath(__file__)))}
    shared = ["--pgn", arg("--pgn", "data/sp2_*.pgn"),
              "--games", str(arg("--games", 2, int)),
              "--sims", str(arg("--sims", 3000, int)),
              "--ckpt", arg("--ckpt", "checkpoints/v2_best.pt")]

    print(f"old code: {os.path.abspath(folders['old'])}")
    print(f"new code: {os.path.abspath(folders['new'])}")
    print(f"{shared[3]} games, {shared[5]} sims a search, {shared[7]}\n")

    results = {}
    for label, key, settings in VARIANTS:
        out = subprocess.run(
            [sys.executable, "-u", __file__, "--code", folders[key],
             "--settings", json.dumps(settings)] + shared,
            capture_output=True, text=True)
        lines = out.stdout.strip().splitlines()
        if out.returncode or not lines:
            print(f"{label:<22} FAILED\n{out.stderr[-1500:]}")
            continue
        r = json.loads(lines[-1])
        results[label] = r
        if len(results) == 1:
            print(f"device {r['device']} ({r['gpu']}) | torch threads "
                  f"{r['threads']} | usable cores {r['cores']}\n")
        base = results.get("current code")
        speedup = f"x{r['rate'] / base['rate']:.2f}" if base else ""
        agree = ""
        if base and r is not base:
            same = sum(a == b for a, b in zip(base["choices"], r["choices"]))
            agree = f"| same move {same}/{len(r['choices'])}"
        print(f"{label:<22} {r['rate']:7.0f} sims/s {speedup:>6} | "
              f"network {r['network']:4.0%} | CPU {r['cpu_us']:6.1f} us/sim "
              f"{agree}")


if __name__ == "__main__":
    main()
