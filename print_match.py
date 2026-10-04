"""
print_match.py - like match.py, but plays one game at a time and prints every
move as it happens, so you can watch instead of waiting for a summary.

Usage:
    python print_match.py --games 4 --nodes 1000 --sims 800
    python print_match.py --games 2 --nodes 5000 --sims 4000 --board
    python print_match.py --games 10 --nodes 325 --sims 400 --pgn watched.pgn

Options:
    --games     number of games, rounded up to an even number    (default 4)
    --sims      simulations per move for your bot              (default 800)
    --nodes     node limit for Stockfish                      (default 1000)
    --ckpt      which checkpoint to play           (default evaltuned_best.pt)
    --cpuct     exploration constant                           (default 2.0)
    --contempt  draw contempt                                    (default 0)
    --material  material weight in leaf evaluations              (default 0)
    --quiesce   plies of capture resolution before evaluating    (default 0)
    --batch     leaves per network call                         (default 48)
    --opening   random plies to start each game from             (default 6)
    --maxmoves  adjudicate as a draw after this many moves     (default 150)
    --board     redraw the board after every move
    --pgn       write all games to this file

Deliberately single-process. Parallel workers would interleave their output
into nonsense, and watching is the whole point here - use match.py when you
want throughput and only care about the final number.

It also uses search_batched, so this is the same search your Lichess bot
runs. match.py uses the older serial search because that parallelises across
CPU cores better, which means the two can disagree slightly.
"""

import inspect
import math
import os
import random
import sys
import time

import torch
import chess
import chess.engine
import chess.pgn

from train import ChessNet
from search_batched import run_search, choose

# Older copies of search_batched.py predate the batch_size and quiesce
# parameters. Check once, so a version mismatch degrades gracefully instead
# of raising TypeError halfway through the first game.
_SEARCH_ACCEPTS = set(inspect.signature(run_search).parameters)


def search(model, board, settings):
    extra = {}
    if "batch_size" in _SEARCH_ACCEPTS:
        extra["batch_size"] = settings["batch"]
    if "quiesce" in _SEARCH_ACCEPTS:
        extra["quiesce"] = settings["quiesce"]
    return run_search(model, board, settings["sims"], settings["contempt"],
                      settings["material"], settings["cpuct"], **extra)


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
    return model, ckpt


def random_opening(plies):
    while True:
        board = chess.Board()
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
        if not board.is_game_over():
            return board


def move_prefix(board):
    """'12.' for White to move, '12...' for Black."""
    number = board.fullmove_number
    return f"{number:>3}." if board.turn == chess.WHITE else f"{number:>3}..."


def play_game(model, engine, bot_is_white, opening, settings, show_board):
    board = opening.copy()
    moves_played = []

    if opening.move_stack:
        opening_line = " ".join(
            m.uci() for m in opening.move_stack)
        print(f"  from random opening: {opening_line}")

    while not board.is_game_over(claim_draw=True):
        if len(board.move_stack) - len(opening.move_stack) >= settings["maxmoves"] * 2:
            print("  adjudicated a draw at the move limit")
            return 0.5, moves_played, "move limit"

        prefix = move_prefix(board)
        bots_turn = board.turn == (chess.WHITE if bot_is_white else chess.BLACK)
        start = time.time()

        if bots_turn:
            root = search(model, board, settings)
            move, ranked = choose(root)
            score = -ranked[0][1].value
            san = board.san(move)
            elapsed = time.time() - start
            alternatives = ", ".join(
                f"{board.san(m)} {int(n.visits)}" for m, n in ranked[1:3])
            print(f"  {prefix} {san:<7} bot   {elapsed:5.1f}s  "
                  f"eval {score:+.2f}   also: {alternatives}")
        else:
            result = engine.play(board,
                                 chess.engine.Limit(nodes=settings["nodes"]))
            move = result.move
            if move is None:
                return 0.5, moves_played, "engine gave no move"
            san = board.san(move)
            elapsed = time.time() - start
            print(f"  {prefix} {san:<7} sf    {elapsed:5.1f}s")

        board.push(move)
        moves_played.append(move)

        if show_board:
            print()
            print(board if bot_is_white else
                  board.mirror().transform(chess.flip_vertical))
            print()

    outcome = board.outcome(claim_draw=True)
    if outcome.winner is None:
        score = 0.5
    elif outcome.winner == (chess.WHITE if bot_is_white else chess.BLACK):
        score = 1.0
    else:
        score = 0.0
    return score, moves_played, outcome.termination.name


def elo_difference(scores):
    n = len(scores)
    mean = sum(scores) / n
    if mean <= 0 or mean >= 1:
        return None, None, None

    def to_elo(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return -400.0 * math.log10(1.0 / p - 1.0)

    variance = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
    stderr = math.sqrt(variance / n)
    return to_elo(mean), to_elo(mean - 1.96 * stderr), to_elo(mean + 1.96 * stderr)


def main():
    games = arg("--games", 4, int)
    games += games % 2
    settings = {
        "sims": arg("--sims", 800, int),
        "nodes": arg("--nodes", 1000, int),
        "cpuct": arg("--cpuct", 2.0, float),
        "contempt": arg("--contempt", 0.0, float),
        "material": arg("--material", 0.0, float),
        "quiesce": arg("--quiesce", 0, int),
        "batch": arg("--batch", 48, int),
        "maxmoves": arg("--maxmoves", 150, int),
    }
    ckpt_path = arg("--ckpt", "checkpoints/evaltuned_best.pt")
    opening_plies = arg("--opening", 6, int)
    pgn_path = arg("--pgn")
    show_board = "--board" in sys.argv

    if not os.path.exists(ckpt_path):
        print(f"no checkpoint at {ckpt_path}")
        sys.exit(1)

    model, ckpt = load_model(ckpt_path)

    try:
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    except FileNotFoundError:
        print("Stockfish not found. Install it with:  brew install stockfish")
        sys.exit(1)
    engine.configure({"Threads": 1, "Hash": 16})

    version = (f"{ckpt.get('blocks', 6)}x{ckpt.get('channels', 128)}"
               f"-s{ckpt.get('step', 0)}-e{ckpt.get('eval_step', 0)}")
    print(f"{ckpt_path}  ({version})")
    unsupported = [name for name in ("batch_size", "quiesce")
                   if name not in _SEARCH_ACCEPTS]
    print(f"{settings['sims']} sims  |  cpuct {settings['cpuct']}  |  "
          f"contempt {settings['contempt']}  |  material {settings['material']}"
          f"  |  quiesce {settings['quiesce']}")
    if unsupported:
        print(f"  note: your search_batched.py does not support "
              f"{', '.join(unsupported)} - those settings are being ignored")
    print(f"versus Stockfish limited to {settings['nodes']:,} nodes")
    print(f"{games} games from {opening_plies}-ply random openings")

    pgn_file = open(pgn_path, "w") if pgn_path else None
    scores = []
    wins = draws = losses = 0

    try:
        for pair in range(games // 2):
            opening = random_opening(opening_plies)

            for bot_is_white in (True, False):
                played = len(scores) + 1
                colour = "White" if bot_is_white else "Black"
                print(f"\n{'-' * 60}")
                print(f"game {played}/{games}  -  bot plays {colour}")
                print(f"{'-' * 60}")

                score, moves, reason = play_game(model, engine, bot_is_white,
                                                 opening, settings, show_board)
                scores.append(score)
                if score == 1.0:
                    wins += 1
                    verdict = "bot wins"
                elif score == 0.5:
                    draws += 1
                    verdict = "draw"
                else:
                    losses += 1
                    verdict = "bot loses"

                pct = (wins + 0.5 * draws) / len(scores) * 100
                print(f"\n  {verdict} ({reason.lower()})   "
                      f"running: +{wins} ={draws} -{losses}  {pct:.1f}%")

                if pgn_file:
                    board = opening.copy()
                    for move in moves:
                        board.push(move)
                    game = chess.pgn.Game.from_board(board)
                    game.headers["White"] = "bot" if bot_is_white else "stockfish"
                    game.headers["Black"] = "stockfish" if bot_is_white else "bot"
                    game.headers["Event"] = (f"{settings['sims']} sims vs "
                                             f"{settings['nodes']} nodes")
                    print(game, file=pgn_file, end="\n\n")
                    pgn_file.flush()

    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        engine.quit()
        if pgn_file:
            pgn_file.close()

    if not scores:
        return

    played = len(scores)
    pct = (wins + 0.5 * draws) / played * 100
    print(f"\n{'=' * 60}")
    print(f"  {played} games:  +{wins} ={draws} -{losses}   ({pct:.1f}%)")

    elo, low, high = elo_difference(scores)
    if elo is None:
        print("  score is 0% or 100% - no Elo estimate possible")
    else:
        print(f"  Elo difference: {elo:+.0f}   "
              f"(95% interval {low:+.0f} to {high:+.0f})")


if __name__ == "__main__":
    main()
