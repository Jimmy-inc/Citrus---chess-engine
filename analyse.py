"""
analyse.py - work out why the bot played a particular move in a position.

Paste in the FEN from a game where it blundered and this shows you the raw
policy priors, the search result, and - if Stockfish is around - what the
refutation was and whether the search ever looked at it.

Usage:
    python analyse.py "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"
    python analyse.py "FEN" --sims 8000 --judge 1000000
    python analyse.py "FEN" --compare

Options:
    --sims      simulations per search                        (default 4000)
    --ckpt      checkpoint to use            (default evaltuned_best.pt)
    --cpuct     exploration constant                          (default 2.0)
    --contempt  draw contempt                                   (default 0)
    --material  material weight in leaf evaluations              (default 0)
    --quiesce   plies of capture resolution                      (default 0)
    --smooth    uniform probability mixed into the policy         (default 0)
    --forced    depth to which every child gets a forced visit    (default 0)
    --batch     leaves per network call                         (default 48)
    --judge     Stockfish nodes for a second opinion        (default 500000)
    --compare   run several settings combinations side by side
    --move      focus on one move (SAN or UCI) and show why it was chosen
    --pv        plies of principal variation to print          (default 6)

--move is how you chase down a specific blunder. Give it the move the bot
actually played and you get its rank, visits and Q, the line the search
believed would follow, and what Stockfish thinks of the position afterwards.
Where those two diverge is the position your value head is misjudging.

The single most useful line in the output is the prior the net gave to
Stockfish's move. If it is near zero, the search never had a chance to
consider it and no amount of extra simulations would have helped - that is
the case --smooth exists for.
"""

import math
import os
import sys

import torch
import torch.nn.functional as F
import chess
import chess.engine

from train import ChessNet
from play import encode_board
from search_batched import (run_search, choose, legal_index_map, device_for,
                            DEFAULT_BATCH)

DEFAULT_CKPT = "checkpoints/evaltuned_best.pt"


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


@torch.no_grad()
def raw_policy(model, board):
    """The net's opinion with no search at all: {move: prior}, plus value."""
    device = device_for(model)
    logits, value = model(encode_board(board).to(device))
    by_index = legal_index_map(board)
    indices = list(by_index.keys())
    priors = F.softmax(logits[0].cpu()[indices], dim=0)
    return ({by_index[i]: float(p) for i, p in zip(indices, priors)},
            value.item())


def principal_variation(board, move, node, plies):
    """
    Follow the most-visited child down, the line the search believed.

    `node` is the node reached *by playing* `move`, so the move has to go on
    the board before its children mean anything - getting that wrong yields
    moves that are illegal in the position they are printed against.
    """
    working = board.copy(stack=False)
    line = [working.san(move)]
    working.push(move)

    current = node
    while len(line) < plies and current.children:
        next_move, child = max(current.children.items(),
                               key=lambda kv: kv[1].visits)
        if child.visits == 0:
            break
        line.append(working.san(next_move))
        working.push(next_move)
        current = child
    return line


def to_centipawns(value):
    value = max(-0.999, min(0.999, value))
    return 400.0 * math.atanh(value)


def stockfish_view(board, nodes):
    """Stockfish's best move and evaluation, or None if it isn't installed."""
    try:
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    except FileNotFoundError:
        return None
    engine.configure({"Threads": 4, "Hash": 512})
    try:
        info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
        score = info["score"].pov(board.turn)
        best = info["pv"][0] if info.get("pv") else None
        cp = None if score.is_mate() else score.score()
        mate = score.mate() if score.is_mate() else None
        return best, cp, mate
    finally:
        engine.quit()


def describe_score(cp, mate):
    if mate is not None:
        return f"#{mate:+d}"
    return f"{cp / 100.0:+.2f}"


def run_one(model, board, settings):
    root = run_search(model, board, settings["sims"], settings["contempt"],
                      settings["material"], settings["cpuct"],
                      batch_size=settings["batch"],
                      quiesce=settings["quiesce"],
                      smoothing=settings["smooth"],
                      forced_depth=settings["forced"])
    move, ranked = choose(root)
    return move, ranked


COMPARISONS = [
    ("baseline",           dict(quiesce=0, smooth=0.0,  material=0.0, forced=0)),
    ("smooth 0.03",        dict(quiesce=0, smooth=0.03, material=0.0, forced=0)),
    ("smooth 0.10",        dict(quiesce=0, smooth=0.10, material=0.0, forced=0)),
    ("material 0.2",       dict(quiesce=0, smooth=0.0,  material=0.2, forced=0)),
    ("quiesce 8",          dict(quiesce=8, smooth=0.0,  material=0.0, forced=0)),
    ("forced 2",           dict(quiesce=0, smooth=0.0,  material=0.0, forced=2)),
    ("smooth + material",  dict(quiesce=0, smooth=0.03, material=0.2, forced=0)),
]


def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    # Skip values that belong to flags.
    flags_with_values = {"--sims", "--ckpt", "--cpuct", "--contempt",
                         "--material", "--quiesce", "--smooth", "--forced",
                         "--batch", "--judge"}
    values = {sys.argv[sys.argv.index(f) + 1]
              for f in flags_with_values if f in sys.argv}
    fens = [a for a in positional if a not in values]

    if not fens:
        print(__doc__)
        sys.exit(1)

    try:
        board = chess.Board(fens[0])
    except ValueError as exc:
        print(f"not a valid FEN: {exc}")
        sys.exit(1)

    ckpt_path = arg("--ckpt", DEFAULT_CKPT)
    if not os.path.exists(ckpt_path):
        print(f"no checkpoint at {ckpt_path}")
        sys.exit(1)

    settings = {
        "sims": arg("--sims", 4000, int),
        "cpuct": arg("--cpuct", 2.0, float),
        "contempt": arg("--contempt", 0.0, float),
        "material": arg("--material", 0.0, float),
        "quiesce": arg("--quiesce", 0, int),
        "smooth": arg("--smooth", 0.0, float),
        "forced": arg("--forced", 0, int),
        "batch": arg("--batch", DEFAULT_BATCH, int),
    }
    judge_nodes = arg("--judge", 500_000, int)

    model, ckpt = load_model(ckpt_path)
    version = (f"{ckpt.get('blocks', 6)}x{ckpt.get('channels', 128)}"
               f"-s{ckpt.get('step', 0)}-e{ckpt.get('eval_step', 0)}")

    print(board)
    print()
    print(f"{'White' if board.turn == chess.WHITE else 'Black'} to move   "
          f"{board.fen()}")
    print(f"{ckpt_path}  ({version})\n")

    priors, raw_value = raw_policy(model, board)
    print(f"net's own evaluation, before any search: "
          f"{raw_value:+.3f}  ({to_centipawns(raw_value)/100:+.2f} pawns)")

    top_priors = sorted(priors.items(), key=lambda kv: kv[1], reverse=True)
    shown = ", ".join(f"{board.san(m)} {p*100:.1f}%" for m, p in top_priors[:5])
    print(f"policy head likes: {shown}\n")

    truth = stockfish_view(board, judge_nodes) if judge_nodes else None
    if truth:
        sf_move, sf_cp, sf_mate = truth
        print(f"Stockfish at {judge_nodes:,} nodes: "
              f"{board.san(sf_move)}  ({describe_score(sf_cp, sf_mate)})")
        rank = [m for m, _ in top_priors].index(sf_move) + 1
        prior = priors[sf_move] * 100
        print(f"  your net rates that move #{rank} of {len(top_priors)}, "
              f"prior {prior:.3f}%")
        if prior < 0.5:
            print("  that is low enough that the search may never try it - "
                  "this is what --smooth is for")
        print()

    if "--compare" in sys.argv:
        print(f"{'settings':<20} {'move':<8} {'visits':>7} {'Q':>7}   "
              f"{'agrees with SF' if truth else ''}")
        print("-" * 60)
        for name, overrides in COMPARISONS:
            trial = dict(settings)
            trial.update(overrides)
            move, ranked = run_one(model, board, trial)
            visits = int(ranked[0][1].visits)
            q = -ranked[0][1].value
            agrees = ""
            if truth:
                agrees = "yes" if move == truth[0] else "no"
            print(f"{name:<20} {board.san(move):<8} {visits:>7} "
                  f"{q:>+7.3f}   {agrees}")
        return

    focus = arg("--move")
    pv_plies = arg("--pv", 6, int)

    move, ranked = run_one(model, board, settings)
    print(f"search with {settings['sims']} sims, cpuct {settings['cpuct']}, "
          f"contempt {settings['contempt']}, material {settings['material']}, "
          f"quiesce {settings['quiesce']}, smooth {settings['smooth']}, "
          f"forced {settings['forced']}\n")

    print(f"{'move':<8} {'visits':>7} {'Q':>8} {'prior':>8}")
    print("-" * 34)
    for m, node in ranked[:8]:
        print(f"{board.san(m):<8} {int(node.visits):>7} "
              f"{-node.value:>+8.3f} {priors.get(m, 0)*100:>7.2f}%")

    print(f"\nit plays {board.san(move)}")
    if truth and move != truth[0]:
        print(f"Stockfish prefers {board.san(truth[0])} - "
              f"look at how many visits that got above")

    print(f"\nlines the search believed:")
    for m, node in ranked[:3]:
        line = principal_variation(board, m, node, pv_plies)
        print(f"  {' '.join(line)}")

    if focus:
        target = None
        for parser in (board.parse_san, board.parse_uci):
            try:
                target = parser(focus)
                break
            except ValueError:
                continue
        if target is None or target not in board.legal_moves:
            print(f"\n{focus} is not a legal move here")
            return

        rank = [m for m, _ in ranked].index(target) + 1
        node = dict(ranked)[target]
        print(f"\n--- {board.san(target)} in detail ---")
        print(f"  ranked #{rank} of {len(ranked)}, "
              f"{int(node.visits)} visits, Q {-node.value:+.3f}, "
              f"prior {priors.get(target, 0)*100:.2f}%")
        print(f"  the search expected: "
              f"{' '.join(principal_variation(board, target, node, pv_plies))}")

        after = board.copy(stack=False)
        after.push(target)
        if judge_nodes:
            reply = stockfish_view(after, judge_nodes)
            if reply:
                reply_move, reply_cp, reply_mate = reply
                # Stockfish scores from the mover's side; flip it so both
                # numbers are from the point of view of whoever moved.
                shown = describe_score(
                    -reply_cp if reply_cp is not None else None,
                    -reply_mate if reply_mate is not None else None)
                print(f"  Stockfish after it: {shown}, "
                      f"answering {after.san(reply_move)}")
                if reply_cp is not None:
                    search_cp = to_centipawns(-node.value)
                    gap = abs(search_cp - (-reply_cp))
                    print(f"  your search said {search_cp/100:+.2f}, "
                          f"Stockfish says {-reply_cp/100:+.2f} "
                          f"- off by {gap/100:.2f}")
                    print(f"\n  next step: run this tool on the position "
                          f"after {board.san(target)} to see whether the net "
                          f"realises it is worse once it gets there.")
                    print(f"  {after.fen()}")


if __name__ == "__main__":
    main()
