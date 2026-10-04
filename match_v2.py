"""
match_v2.py - play the version two net against Stockfish and report a result.

Same idea as match.py, but driving search_v2 so it works with the canonical
encoding and the convolutional policy head. Single process, so it is slower
than match.py but it measures the engine you would actually deploy.

Requires Stockfish on your PATH:  brew install stockfish

Usage:
    python match_v2.py --games 20 --nodes 1000 --sims 800
    python match_v2.py --games 100 --nodes 325 --sims 400 --quiet
    python match_v2.py --games 4 --nodes 10000 --sims 4000 --judge 200000

Options:
    --games     games to play, rounded up to an even number    (default 20)
    --sims      simulations per move                          (default 800)
    --nodes     node limit for Stockfish - the strength dial  (default 1000)
    --ckpt      checkpoint to play              (default checkpoints/v2_best)
    --cpuct     exploration constant                          (default 2.0)
    --contempt  draw contempt                                   (default 0)
    --material  material weight in leaf evaluations             (default 0)
    --quiesce   plies of quiescence search                      (default 0)
    --batch     leaves per network call                        (default 48)
    --opening   random plies to start each game from            (default 6)
    --seed      fix the random openings; give matches comparing
                different nets the same seed                 (default none)
    --claim-draws  search with the old draw rule, where a position counts
                as drawn if any move could claim a draw - for comparing
                against the current rule
    --fpu       first-play urgency reduction, e.g. 0.3; leave out for
                the original rule (unsearched moves count as equal)
    --old-castling  look castling up at e1g1, the old lookup the nets
                were never trained on - only for measuring the fix
    --maxmoves  adjudicate a draw after this many moves        (default 150)
    --judge     Stockfish nodes for a second opinion per move    (default 0)
    --quiet     only print one line per game, not per move
    --pgn       write the games to this file

Node limiting is an honest handicap - Stockfish still plays the best move it
found, it just searched less. That is not the same as Skill Level, which
makes it choose worse moves deliberately.
"""

import math
import os
import random
import sys
import time

import torch
import chess
import chess.engine
import chess.pgn

import encoding_v2
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


def move_prefix(board):
    number = board.fullmove_number
    return f"{number:>3}." if board.turn == chess.WHITE else f"{number:>3}..."


def play_game(model, engine, bot_is_white, opening, settings, quiet):
    board = opening.copy()
    moves_played = []
    bot_colour = chess.WHITE if bot_is_white else chess.BLACK

    while not board.is_game_over(claim_draw=True):
        played = len(board.move_stack) - len(opening.move_stack)
        if played >= settings["maxmoves"] * 2:
            return 0.5, moves_played, "move limit"

        prefix = move_prefix(board)
        start = time.time()

        if board.turn == bot_colour:
            root = engine_v2.run_search(
                model, board, settings["sims"], settings["contempt"],
                settings["material"], settings["cpuct"],
                batch_size=settings["batch"], quiesce=settings["quiesce"],
                syzygy=settings["syzygy"], fpu=settings["fpu"])
            move, ranked = engine_v2.choose(root)
            score = -ranked[0][1].value
            san = board.san(move)
            elapsed = time.time() - start
            line = (f"  {prefix} {san:<7} bot   {elapsed:5.1f}s  "
                    f"bot {to_centipawns(score)/100:+.2f}")
        else:
            result = engine.play(board,
                                 chess.engine.Limit(nodes=settings["nodes"]))
            move = result.move
            if move is None:
                return 0.5, moves_played, "engine gave no move"
            san = board.san(move)
            elapsed = time.time() - start
            line = f"  {prefix} {san:<7} sf    {elapsed:5.1f}s"

        board.push(move)
        moves_played.append(move)

        if settings["judge"]:
            info = engine.analyse(
                board, chess.engine.Limit(nodes=settings["judge"]))
            pov = info["score"].pov(bot_colour)
            if pov.is_mate():
                line += f"   sf #{pov.mate():+d}"
            else:
                line += f"   sf {pov.score()/100:+.2f}"

        if not quiet:
            print(line)

    outcome = board.outcome(claim_draw=True)
    if outcome.winner is None:
        score = 0.5
    elif outcome.winner == bot_colour:
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
    games = arg("--games", 20, int)
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
        "judge": arg("--judge", 0, int),
        "syzygy": arg("--syzygy", "syzygy"),
        "fpu": arg("--fpu", None, float),
    }
    ckpt_path = arg("--ckpt", "checkpoints/v2_best.pt")
    opening_plies = arg("--opening", 6, int)
    pgn_path = arg("--pgn")
    # Matches comparing different nets should share a seed, so every net
    # meets the same openings and the difference is the net, not the draw.
    seed = arg("--seed", None, int)
    if seed is not None:
        random.seed(seed)
    engine_v2.CLAIM_DRAWS = "--claim-draws" in sys.argv
    # The fix is the default; --castling-fix is still accepted, and changes
    # nothing. --old-castling plays with the old, broken lookup.
    encoding_v2.CASTLE_ONTO_ROOK = "--old-castling" not in sys.argv
    quiet = "--quiet" in sys.argv

    if not os.path.exists(ckpt_path):
        print(f"no checkpoint at {ckpt_path}")
        sys.exit(1)

    model, ckpt = engine_v2.load(ckpt_path)
    device = engine_v2.device_for(model)

    try:
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    except FileNotFoundError:
        print("Stockfish not found. Install it with:  brew install stockfish")
        sys.exit(1)
    engine.configure({"Threads": 1, "Hash": 16})

    print(f"{ckpt_path}  ({ckpt.get('blocks', 12)}x{ckpt.get('channels', 256)}"
          f", step {ckpt.get('step', 0):,}) on {device}")
    print(f"{settings['sims']} sims  |  cpuct {settings['cpuct']}  |  "
          f"contempt {settings['contempt']}  |  material {settings['material']}"
          f"  |  quiesce {settings['quiesce']}  |  fpu "
          f"{'off' if settings['fpu'] is None else settings['fpu']}")
    print(f"versus Stockfish limited to {settings['nodes']:,} nodes")
    print("draws in search: "
          + ("claimable counts (old rule)" if engine_v2.CLAIM_DRAWS
             else "only once they happen"))
    print("castling looked up: "
          + ("king onto rook, as trained (fixed)"
             if encoding_v2.CASTLE_ONTO_ROOK else "e1g1 (the old lookup)"))
    print(f"{games} games from {opening_plies}-ply random openings"
          + (f", seed {seed}" if seed is not None else "") + "\n")

    pgn_file = open(pgn_path, "w") if pgn_path else None
    scores = []
    wins = draws = losses = 0
    start = time.time()

    try:
        for pair in range(games // 2):
            opening = random_opening(opening_plies)

            for bot_is_white in (True, False):
                played = len(scores) + 1
                colour = "White" if bot_is_white else "Black"
                if not quiet:
                    print(f"\n{'-' * 58}")
                    print(f"game {played}/{games}  -  bot plays {colour}")
                    print(f"{'-' * 58}")

                score, moves, reason = play_game(model, engine, bot_is_white,
                                                 opening, settings, quiet)
                scores.append(score)
                if score == 1.0:
                    wins += 1
                    verdict = "win "
                elif score == 0.5:
                    draws += 1
                    verdict = "draw"
                else:
                    losses += 1
                    verdict = "loss"

                pct = (wins + 0.5 * draws) / len(scores) * 100
                elapsed = time.time() - start
                print(f"game {played:>3}/{games}  bot as {colour:<5} "
                      f"{verdict}  ({reason.lower()})  |  "
                      f"+{wins} ={draws} -{losses}  {pct:5.1f}%  |  "
                      f"{elapsed/played:.0f}s/game")

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
    print(f"\n{'=' * 58}")
    print(f"  {played} games:  +{wins} ={draws} -{losses}   ({pct:.1f}%)")

    elo, low, high = elo_difference(scores)
    if elo is None:
        print("  scored 0% or 100% - change the node limit for a useful number")
    else:
        print(f"  Elo difference: {elo:+.0f}   "
              f"(95% interval {low:+.0f} to {high:+.0f})")
        print(f"  positive means stronger than Stockfish at "
              f"{settings['nodes']:,} nodes")
        if high - low > 150:
            print("\n  That interval is wide - play more games.")


if __name__ == "__main__":
    main()
