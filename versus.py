"""
versus.py - two versions of the bot play each other: the same net with
different search settings, or two different nets.

Usage:
    python versus.py --a "fpu=0.3" --b "" --games 400 --sims 800 --seed 1
    python versus.py --a "cpuct=2.5" --b "cpuct=2.0" --games 400 --seed 1
    python versus.py --ckpt-a checkpoints/new_best.pt \\
                     --ckpt-b checkpoints/mix_best.pt --games 400 --seed 1

This is the usual way to tune search settings: a head-to-head result
measures the difference directly, where two separate matches against
Stockfish each carry their own noise. Two cautions:

  * self-play exaggerates differences - a setting worth +40 here may be
    worth +20 against anyone else - and can occasionally reward a quirk
    that only beats its sibling. Confirm the final choice against
    Stockfish with match_v2.py before trusting it.
  * both sides search a fixed number of simulations, which is fair for
    settings that do not change speed (fpu, cpuct). A change that makes
    the search slower would get its extra cost for free here.

Each random opening is played twice with colours swapped, as in
match_v2.py, so neither side gets the better openings. Give comparisons the
same --seed so they meet the same openings.

Settings for each side, comma separated name=value, empty for defaults:
    cpuct     exploration constant                           (default 2.0)
    fpu       first-play urgency reduction, or off           (default off)
    sims      simulations a move                (default --sims, below)
    castling  new, or old for the broken lookup              (default new)
    contempt  draw contempt                                  (default 0.0)
    material  material weight in leaf evaluations            (default 0.0)
    quiesce   plies of quiescence search                       (default 0)

Options:
    --a, --b          settings for each side                 (default "")
    --ckpt            checkpoint for both    (default checkpoints/mix_best.pt)
    --ckpt-a/--ckpt-b a different checkpoint for one side
    --games           games, rounded up to even                 (default 100)
    --sims            simulations a move unless a side says   (default 800)
    --opening         random plies to start from                  (default 6)
    --seed            fix the openings                         (default none)
    --maxmoves        moves a side before calling it a draw     (default 150)
    --pgn             write the games here
"""

import os
import random
import sys
import time

import chess
import chess.pgn

import encoding_v2
import search_v2 as engine_v2
from match_v2 import elo_difference, random_opening

KNOWN = {"cpuct", "fpu", "sims", "castling", "contempt", "material",
         "quiesce"}


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def parse_side(text, sims):
    """A side's settings, refusing names it does not know - a typo should
    stop the run, not silently play the default."""
    side = {"cpuct": engine_v2.C_PUCT, "fpu": None, "sims": sims,
            "castling": "new", "contempt": engine_v2.DRAW_CONTEMPT,
            "material": engine_v2.MATERIAL_WEIGHT, "quiesce": 0}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, _, value = part.partition("=")
        name, value = name.strip(), value.strip()
        if name not in KNOWN or not value:
            sys.exit(f"don't understand {part!r} - settings are "
                     f"{', '.join(sorted(KNOWN))}, as name=value")
        if name == "fpu":
            side[name] = None if value == "off" else float(value)
        elif name == "castling":
            if value not in ("new", "old"):
                sys.exit("castling is new or old")
            side[name] = value
        elif name in ("sims", "quiesce"):
            side[name] = int(value)
        else:
            side[name] = float(value)
    return side


def describe(side, defaults):
    changed = [f"{k}={'off' if side[k] is None else side[k]}"
               for k in sorted(side) if side[k] != defaults[k]]
    return ", ".join(changed) or "defaults"


def think(model, board, side):
    encoding_v2.CASTLE_ONTO_ROOK = side["castling"] == "new"
    root = engine_v2.run_search(
        model, board, side["sims"], side["contempt"], side["material"],
        side["cpuct"], quiesce=side["quiesce"], fpu=side["fpu"])
    return engine_v2.choose(root)[0]


def play_game(players, a_is_white, opening, maxmoves):
    """Score for side A, the moves, and why the game ended."""
    board = opening.copy()
    start = len(board.move_stack)
    engine_v2.clear_cache()                      # each game starts afresh
    while not board.is_game_over(claim_draw=True):
        if len(board.move_stack) - start >= maxmoves * 2:
            return 0.5, board, "move limit"
        a_to_move = (board.turn == chess.WHITE) == a_is_white
        model, side = players["a" if a_to_move else "b"]
        board.push(think(model, board, side))

    outcome = board.outcome(claim_draw=True)
    if outcome.winner is None:
        return 0.5, board, outcome.termination.name.lower()
    a_won = (outcome.winner == chess.WHITE) == a_is_white
    return (1.0 if a_won else 0.0), board, outcome.termination.name.lower()


def main():
    games = arg("--games", 100, int)
    games += games % 2
    sims = arg("--sims", 800, int)
    maxmoves = arg("--maxmoves", 150, int)
    opening_plies = arg("--opening", 6, int)
    seed = arg("--seed", None, int)
    if seed is not None:
        random.seed(seed)

    defaults = parse_side("", sims)
    side_a = parse_side(arg("--a", ""), sims)
    side_b = parse_side(arg("--b", ""), sims)
    ckpt = arg("--ckpt", "checkpoints/mix_best.pt")
    paths = {"a": arg("--ckpt-a", ckpt), "b": arg("--ckpt-b", ckpt)}
    for path in set(paths.values()):
        if not os.path.exists(path):
            sys.exit(f"no checkpoint at {path}")
    if side_a == side_b and paths["a"] == paths["b"]:
        sys.exit("both sides are identical - every game would be a mirror")

    models = {}
    for path in set(paths.values()):
        model, _ = engine_v2.load(path)
        model.to(engine_v2.pick_device())
        models[path] = model
    players = {"a": (models[paths["a"]], side_a),
               "b": (models[paths["b"]], side_b)}

    print(f"A: {paths['a']}  |  {describe(side_a, defaults)}")
    print(f"B: {paths['b']}  |  {describe(side_b, defaults)}")
    print(f"on {engine_v2.device_for(models[paths['a']])}  |  {sims} sims "
          f"unless a side says otherwise")
    print(f"{games} games from {opening_plies}-ply random openings"
          + (f", seed {seed}" if seed is not None else "")
          + ", each opening played with both colours\n", flush=True)

    pgn_path = arg("--pgn")
    pgn_file = open(pgn_path, "w") if pgn_path else None
    names = {"a": f"A ({describe(side_a, defaults)})",
             "b": f"B ({describe(side_b, defaults)})"}
    scores = []
    wins = draws = losses = 0
    began = time.time()

    try:
        for _ in range(games // 2):
            opening = random_opening(opening_plies)
            for a_is_white in (True, False):
                score, board, reason = play_game(players, a_is_white,
                                                 opening, maxmoves)
                scores.append(score)
                wins += score == 1.0
                draws += score == 0.5
                losses += score == 0.0
                played = len(scores)
                verdict = {1.0: "A wins", 0.5: "draw  ", 0.0: "B wins"}[score]
                print(f"game {played:>4}/{games}  A as "
                      f"{'White' if a_is_white else 'Black'}  {verdict}  "
                      f"({reason})  |  A +{wins} ={draws} -{losses}  "
                      f"{(wins + 0.5 * draws) / played:6.1%}  |  "
                      f"{(time.time() - began) / played:.0f}s/game",
                      flush=True)

                if pgn_file:
                    game = chess.pgn.Game.from_board(board)
                    game.headers["White"] = names["a" if a_is_white else "b"]
                    game.headers["Black"] = names["b" if a_is_white else "a"]
                    game.headers["Event"] = "versus.py"
                    print(game, file=pgn_file, end="\n\n")
                    pgn_file.flush()
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        if pgn_file:
            pgn_file.close()

    if not scores:
        return
    played = len(scores)
    print(f"\n{'=' * 58}")
    print(f"  {played} games, A's view:  +{wins} ={draws} -{losses}   "
          f"({(wins + 0.5 * draws) / played:.1%})")
    elo, low, high = elo_difference(scores)
    if elo is None:
        print("  one side scored 0% or 100%")
    else:
        print(f"  A minus B: {elo:+.0f} Elo   (95% interval {low:+.0f} to "
              f"{high:+.0f})")
        if low > 0:
            print("  A is stronger.")
        elif high < 0:
            print("  B is stronger.")
        else:
            print("  Not separated yet - the interval includes zero.")


if __name__ == "__main__":
    main()
