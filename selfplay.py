"""
selfplay.py - play the net against itself and write the games out, keeping
the net's own evaluation of every position alongside the moves.

Usage:
    python selfplay.py --games 200 --pgn data/selfplay.pgn
    python selfplay.py --games 50 --sims 800 --ckpt checkpoints/v2_best.pt
    python selfplay.py --games 400 --opening 8 --pgn data/selfplay.pgn

This is the first half of weakness detection. It only generates games and
makes no judgement about which moves were bad - label_weak.py does that,
reading the PGN written here. Keeping the halves apart matters because
self-play is GPU bound at roughly a hundred games an hour while labelling
is CPU bound and finishes in minutes, so a threshold can be changed and the
same games relabelled rather than replayed.

Both sides are the same net and the search is deterministic, so every game
would otherwise be identical. The random opening plies supply the variety,
the same trick match_v2.py uses. Real variety in the middlegame would need
sampling at the search root, which search_v2.py does not currently do.

Each move's comment carries the net's evaluation as [%eval ...], the usual
PGN convention: White's point of view, in pawns, describing the position
*after* the move. label_weak.py compares that against Stockfish without
having to re-run the net.

Every move in the PGN is a move the net chose. The random opening is stored
as the game's starting position instead of being replayed as moves, so the
labeller never mistakes a random ply for something the net decided to play.

Options:
    --games     games to play                                 (default 20)
    --pgn       where to write them                (default selfplay.pgn)
    --ckpt      checkpoint to play        (default checkpoints/v2_best.pt)
    --sims      simulations a move                           (default 800)
    --opening   random plies before the net takes over          (default 6)
    --maxmoves  moves a side before calling it a draw          (default 150)
    --cpuct     exploration constant                         (default 2.0)
    --contempt  draw contempt                                (default 0.0)
    --material  material nudge                               (default 0.0)
    --quiesce   quiescence depth                               (default 0)
    --batch     evaluation batch size                         (default 48)
    --syzygy    tablebase directory                      (default syzygy)
    --quiet     only print the per-game summary lines
    --force     overwrite the PGN if it already exists

Finished games are flushed as they are played, so cancelling a run keeps
everything except the game in progress. Restarting into the same --pgn
would throw those away, which is why an existing file has to be replaced
deliberately. label_weak.py takes a glob, so leaving each run in its own
file and labelling them together is the easier habit.
"""

import math
import os
import random
import sys
import time

import chess
import chess.pgn

import search_v2 as engine_v2


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def random_opening(plies):
    """A short random legal opening, so the games are not all identical."""
    while True:
        board = chess.Board()
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
        if not board.is_game_over():
            return board


def to_centipawns(value):
    value = max(-0.999, min(0.999, value))
    return 400.0 * math.atanh(value)


def play_game(model, opening, settings):
    """One game, returned as a PGN game with the net's evals in comments."""
    board = opening.copy()
    game = chess.pgn.Game()
    game.setup(opening)
    node = game

    while not board.is_game_over(claim_draw=True):
        if len(board.move_stack) - len(opening.move_stack) >= settings["maxmoves"] * 2:
            return game, "move limit"

        root = engine_v2.run_search(
            model, board, settings["sims"], settings["contempt"],
            settings["material"], settings["cpuct"],
            batch_size=settings["batch"], quiesce=settings["quiesce"],
            syzygy=settings["syzygy"])
        move, ranked = engine_v2.choose(root)

        # choose() ranks children, whose values are from the opponent's view;
        # negating gives the mover's view of the position after the move.
        mover_cp = to_centipawns(-ranked[0][1].value)
        white_cp = mover_cp if board.turn == chess.WHITE else -mover_cp

        board.push(move)
        node = node.add_variation(move)
        node.comment = f"[%eval {white_cp / 100:.2f}]"

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return game, "unfinished"
    return game, outcome.result()


def main():
    settings = {
        "sims": arg("--sims", 800, int),
        "cpuct": arg("--cpuct", 2.0, float),
        "contempt": arg("--contempt", 0.0, float),
        "material": arg("--material", 0.0, float),
        "quiesce": arg("--quiesce", 0, int),
        "batch": arg("--batch", 48, int),
        "maxmoves": arg("--maxmoves", 150, int),
        "syzygy": arg("--syzygy", "syzygy"),
    }
    games = arg("--games", 20, int)
    ckpt_path = arg("--ckpt", "checkpoints/v2_best.pt")
    opening_plies = arg("--opening", 6, int)
    pgn_path = arg("--pgn", "selfplay.pgn")
    quiet = "--quiet" in sys.argv

    if not os.path.exists(ckpt_path):
        print(f"no checkpoint at {ckpt_path}")
        sys.exit(1)

    # Checked before the model loads, so a mistake costs no time at all.
    # An empty file is not worth protecting: a task killed before it
    # finished its first game leaves one behind, and refusing to reuse that
    # name would only make restarting the run harder for no gain.
    if (os.path.exists(pgn_path) and os.path.getsize(pgn_path) > 0
            and "--force" not in sys.argv):
        print(f"{pgn_path} already exists, and opening it would throw away "
              f"the games in it.\nUse another name - label_weak.py takes a "
              f"glob, so selfplay1.pgn and selfplay2.pgn label together - "
              f"or pass --force.")
        sys.exit(1)

    if opening_plies < 2:
        print("warning: fewer than 2 random opening plies means the games "
              "will be near-identical, because the search is deterministic")

    model, ckpt = engine_v2.load(ckpt_path)
    device = engine_v2.device_for(model)

    print(f"{ckpt_path}  ({ckpt.get('blocks', 12)}x{ckpt.get('channels', 256)}"
          f", step {ckpt.get('step', 0):,}) on {device}")
    print(f"{settings['sims']} sims  |  cpuct {settings['cpuct']}  |  "
          f"contempt {settings['contempt']}  |  material {settings['material']}"
          f"  |  quiesce {settings['quiesce']}")
    print(f"{games} self-play games from {opening_plies}-ply random openings")
    print(f"writing {pgn_path}\n")

    results = {}
    plies = 0
    start = time.time()

    with open(pgn_path, "w") as out:
        for number in range(1, games + 1):
            opening = random_opening(opening_plies)
            game, result = play_game(model, opening, settings)

            game.headers["Event"] = "self-play weakness detection"
            game.headers["White"] = game.headers["Black"] = "net"
            game.headers["Round"] = str(number)
            game.headers["Result"] = result if "-" in result else "*"
            game.headers["Annotator"] = (
                f"{ckpt_path} step {ckpt.get('step', 0)} "
                f"{settings['sims']} sims")

            print(game, file=out, end="\n\n")
            out.flush()

            moves = len(list(game.mainline_moves()))
            plies += moves
            results[result] = results.get(result, 0) + 1

            if not quiet:
                elapsed = time.time() - start
                rate = number / (elapsed / 3600) if elapsed else 0
                print(f"  game {number:>4}/{games}  {moves:>3} moves  "
                      f"{result:<10} {rate:5.1f} games/h")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  games        : {games:,}")
    print(f"  net moves    : {plies:,}")
    print(f"  rate         : {games/(elapsed/3600):.0f} games/h")
    for result, count in sorted(results.items()):
        print(f"  {result:<12} : {count:,}")
    print(f"\nNow label them:  python label_weak.py --pgn {pgn_path} "
          f"--out data/weakness.bin")


if __name__ == "__main__":
    main()
