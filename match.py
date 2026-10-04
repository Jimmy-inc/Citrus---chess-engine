"""
match.py - play your bot against Stockfish automatically and report the result.

Requires Stockfish on your PATH:  brew install stockfish

Usage:
    python match.py --games 40 --nodes 10000 --sims 400
    python match.py --games 100 --nodes 1000 --sims 200 --ckpt checkpoints/best.pt
    python match.py --games 20 --nodes 100000 --sims 1600 --pgn games.pgn
    python match.py --games 200 --nodes 10000 --sims 400 --workers 8

Options:
    --games     number of games, rounded up to an even number   (default 20)
    --sims      simulations per move for your bot               (default 400)
    --nodes     node limit for Stockfish - this is the dial     (default 10000)
    --ckpt      which checkpoint to play                (default evaltuned_best)
    --cpuct     exploration constant                            (default 2.0)
    --contempt  draw contempt                                   (default 0.3)
    --material  material weight in leaf evaluations             (default 0.2)
    --opening   random plies to start each game from            (default 6)
    --maxmoves  adjudicate as a draw after this many moves       (default 150)
    --pgn       write all games to this file
    --workers   number of games to run in parallel              (default 2)
    --engine-threads  Threads given to each Stockfish instance   (default 1)
    --engine-hash     Hash MB given to each Stockfish instance   (default 16)

Games are played in colour-reversed pairs from the same random opening, so
neither side gets an easy run of White. Each pair is dispatched as one job to
a worker process, which owns a persistent model copy and Stockfish instance
for the life of the run - so parallelism comes from playing many games at
once, not from re-loading anything per game.

def test: python match.py --games 500 --nodes 1000 --sims 800 --ckpt checkpoints/evaltuned_best.pt --cpuct 2 --contempt 0 --material 0

Node limiting is an honest handicap: Stockfish still plays the best move it
found, it just searched less. That is different from Skill Level, which makes
it choose worse moves on purpose.

A note on --workers: each worker process pins torch to a single thread (see
_init_worker) before doing anything else. Without that, PyTorch's default
intra-op threading means every worker independently tries to use *all* your
cores for its (tiny, single-position) forward passes, and N workers doing
that fight each other instead of adding up. Capping each worker to one
thread and letting the process pool be the source of parallelism is what
actually uses the extra cores.

GPU note: load_model() never moves the model onto a CUDA device, so even
with a GPU present this runs on CPU only. Parallelizing across processes
does not change that - getting the GPU meaningfully busy would mean
batching leaf evaluations across many in-flight searches at once, which
means restructuring run_search()/choose() in search.py to yield eval
requests instead of blocking on them. Worth doing if CPU-parallel still
isn't fast enough for you, but it's a separate, bigger change.
"""

import math
import multiprocessing as mp
import os
import random
import sys
import time

import torch
import chess
import chess.engine
import chess.pgn

from train import ChessNet
from search import run_search, choose


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = ChessNet(blocks=ckpt.get("blocks", 6),
                     channels=ckpt.get("channels", 128))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("step", 0)


def random_opening(plies):
    """A short random legal opening, so games are not all identical."""
    while True:
        board = chess.Board()
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
        if not board.is_game_over():
            return board


def play_game(model, engine, bot_is_white, opening, settings):
    board = opening.copy()

    moves_played = []
    while not board.is_game_over(claim_draw=True):
        if len(board.move_stack) - len(opening.move_stack) >= settings["maxmoves"] * 2:
            return 0.5, moves_played, "move limit"

        if board.turn == (chess.WHITE if bot_is_white else chess.BLACK):
            root = run_search(model, board, settings["sims"],
                              settings["contempt"], settings["material"],
                              settings["cpuct"])
            move, _ = choose(root)
        else:
            result = engine.play(board, chess.engine.Limit(nodes=settings["nodes"]))
            move = result.move
            if move is None:
                return 0.5, moves_played, "engine gave no move"

        board.push(move)
        moves_played.append(move)

    outcome = board.outcome(claim_draw=True)
    if outcome.winner is None:
        score = 0.5
    elif outcome.winner == (chess.WHITE if bot_is_white else chess.BLACK):
        score = 1.0
    else:
        score = 0.0
    return score, moves_played, outcome.termination.name


def elo_difference(scores):
    """Elo gap implied by the score, with a rough 95% interval."""
    n = len(scores)
    mean = sum(scores) / n
    if mean <= 0:
        return None, None, None
    if mean >= 1:
        return None, None, None

    def to_elo(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return -400.0 * math.log10(1.0 / p - 1.0)

    variance = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
    stderr = math.sqrt(variance / n)
    return to_elo(mean), to_elo(mean - 1.96 * stderr), to_elo(mean + 1.96 * stderr)


# --- worker process state -------------------------------------------------
# Each pool worker loads the model and opens Stockfish exactly once, then
# reuses both across every pair it's handed. Living at module scope like
# this is what lets the "spawn"-started child processes pick them up.

_worker_model = None
_worker_engine = None


def _init_worker(ckpt_path, engine_threads, engine_hash):
    global _worker_model, _worker_engine

    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # main process handles Ctrl-C

    # Cap torch to one thread per worker - see the module docstring. Without
    # this, every worker tries to use every core for its own tiny forward
    # passes and they end up contending instead of adding throughput.
    torch.set_num_threads(1)

    # Forked/spawned processes can otherwise inherit correlated RNG state;
    # reseed from OS entropy so openings actually differ across workers.
    random.seed()

    _worker_model, _ = load_model(ckpt_path)
    _worker_engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    _worker_engine.configure({"Threads": engine_threads, "Hash": engine_hash})


def _play_pair(args):
    """Play one colour-reversed pair of games from a fresh random opening."""
    settings, opening_plies, want_pgn = args
    opening = random_opening(opening_plies)

    results = []
    for bot_is_white in (True, False):
        score, moves, reason = play_game(_worker_model, _worker_engine, bot_is_white,
                                         opening, settings)
        pgn_text = None
        if want_pgn:
            board = opening.copy()
            for move in moves:
                board.push(move)
            game = chess.pgn.Game.from_board(board)
            game.headers["White"] = "bot" if bot_is_white else "stockfish"
            game.headers["Black"] = "stockfish" if bot_is_white else "bot"
            game.headers["Event"] = f"{settings['sims']} sims vs " \
                                    f"{settings['nodes']} nodes"
            pgn_text = str(game)
        results.append((score, reason, bot_is_white, pgn_text))
    return results


def main():
    games = arg("--games", 20, int)
    games += games % 2                      # round up to a whole number of pairs
    settings = {
        "sims": arg("--sims", 400, int),
        "nodes": arg("--nodes", 10_000, int),
        "cpuct": arg("--cpuct", 2.0, float),
        "contempt": arg("--contempt", 0.3, float),
        "material": arg("--material", 0.2, float),
        "maxmoves": arg("--maxmoves", 150, int),
    }
    ckpt_path = arg("--ckpt", "checkpoints/evaltuned_best.pt")
    opening_plies = arg("--opening", 6, int)
    pgn_path = arg("--pgn")
    workers = arg("--workers", 2, int)
    engine_threads = arg("--engine-threads", 1, int)
    engine_hash = arg("--engine-hash", 16, int)

    if not os.path.exists(ckpt_path):
        print(f"no checkpoint at {ckpt_path}")
        sys.exit(1)

    # Peek at the step count for the log line without holding onto a full
    # model in the main process - each worker loads its own copy anyway.
    step = torch.load(ckpt_path, map_location="cpu").get("step", 0)

    pairs = games // 2

    print(f"{ckpt_path} (step {step:,}) at {settings['sims']} sims")
    print(f"versus Stockfish limited to {settings['nodes']:,} nodes")
    print(f"{games} games from {opening_plies}-ply random openings")
    print(f"{workers} worker process{'es' if workers != 1 else ''} in parallel\n")

    pgn_file = open(pgn_path, "w") if pgn_path else None
    scores = []
    wins = draws = losses = 0
    start = time.time()

    # "spawn" avoids handing each worker a forked copy of an already-running
    # Stockfish/torch process, and sidesteps the RNG-inheritance gotcha too.
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=workers, initializer=_init_worker,
                     initargs=(ckpt_path, engine_threads, engine_hash))

    job = (settings, opening_plies, pgn_file is not None)
    jobs = [job] * pairs

    try:
        for pair_result in pool.imap_unordered(_play_pair, jobs):
            for score, reason, bot_is_white, pgn_text in pair_result:
                scores.append(score)
                if score == 1.0:
                    wins += 1
                elif score == 0.5:
                    draws += 1
                else:
                    losses += 1

                played = len(scores)
                pct = (wins + 0.5 * draws) / played * 100
                elapsed = time.time() - start
                print(f"game {played:>3}/{games}  "
                      f"bot as {'White' if bot_is_white else 'Black'}  "
                      f"{'win ' if score == 1 else 'draw' if score == .5 else 'loss'}"
                      f"  ({reason.lower()})  |  "
                      f"+{wins} ={draws} -{losses}  {pct:5.1f}%  |  "
                      f"{elapsed/played:.1f}s/game avg ({workers}x parallel)")

                if pgn_file and pgn_text:
                    print(pgn_text, file=pgn_file, end="\n\n")
                    pgn_file.flush()

        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("\nstopped early")
        pool.terminate()
        pool.join()
    finally:
        if pgn_file:
            pgn_file.close()

    if not scores:
        return

    played = len(scores)
    pct = (wins + 0.5 * draws) / played * 100
    print(f"\n{'='*54}")
    print(f"  {played} games:  +{wins} ={draws} -{losses}   ({pct:.1f}%)")

    elo, low, high = elo_difference(scores)
    if elo is None:
        print("  score is 0% or 100% - no Elo estimate possible, "
              "change the node limit and try again")
    else:
        print(f"  Elo difference: {elo:+.0f}   "
              f"(95% interval {low:+.0f} to {high:+.0f})")
        print(f"  positive means your bot is stronger than Stockfish "
              f"at {settings['nodes']:,} nodes")

    if high is not None and high - low > 150:
        print("\n  That interval is wide - play more games for a firmer number.")


if __name__ == "__main__":
    main()
