"""
label_all.py - label every position in self-play games once, keeping
everything Stockfish says about it, so the rule for what counts as a
weakness is chosen afterwards instead of baked into a week of labelling.

label_weak.py decides while it labels and keeps only what it flagged, so a
different threshold means running Stockfish again. This keeps, for every
position: Stockfish's score and best move, the move the net actually
played scored by its own search at the same depth, and the net's own
evaluation. Thresholds are then a few seconds of arithmetic.

Two steps:

    # Stockfish over every position (CPU; split with --shard, resumable)
    python label_all.py label --pgn 'data/sp4_*.pgn' --shard 0/12 \\
        --out data/labels_sp4_0.bin --workers 6 --engine ./stockfish19

    # choose a rule and write training records (seconds; without --out it
    # only reports)
    python label_all.py select --labels 'data/labels_sp*.bin'
    python label_all.py select --labels 'data/labels_sp*.bin' \\
        --out data/weak_v3.bin
    python label_all.py select --labels 'data/labels_sp*.bin' --rule old \\
        --out data/weak_v3_oldrule.bin

The new rule (the default):

  mistake     the played move is worse than Stockfish's best by more than
              --mistake-cp, both searched from the same position at the same
              depth - so Stockfish's drift between two separate searches
              does not count as the net's mistake. No cap: +10 to +8 counts
              like +1 to -1.
  conversion  Stockfish sees a forced mate and the played move does not
              keep it as short (mate in 8, played a move that is mate in 9 -
              the endless-checks case), or loses the mate altogether.
  misjudged   the net's evaluation is more than --misjudged away from
              Stockfish's on the net's own -1..+1 scale, where +8 and +10
              are both about 0.99 rather than 200cp apart.

The old rule reproduces label_weak.py exactly (eval before vs after the
move, capped at +/-1000cp, 50cp and 100cp), so both datasets can be cut from
the same labels and compared.

A record is written as each game finishes, so a killed job keeps its work;
run the same command again and it carries on from the next game.

Options (label):
    --pgn        games to read; quote globs                     (required)
    --out        label file for this shard                      (required)
    --depth      Stockfish depth                                (default 18)
    --workers    parallel Stockfish processes                    (default 4)
    --engine     Stockfish binary                       (default stockfish)
    --hash       hash MB per Stockfish process                (default 128)
    --shard      i/n: this job takes every nth game             (default 0/1)
    --games      stop after this many games                   (default all)

Options (select):
    --labels     label files; quote globs                       (required)
    --out        training records to write; without it, report only
    --rule       new or old                                   (default new)
    --mistake-cp new rule: played move this much worse is a mistake (20)
    --mate-slack new rule: allowed extra moves to mate before a slower mate
                 counts                                          (default 0)
    --misjudged  new rule: value gap on the -1..+1 scale        (default 0.25)
    --drop, --disagree, --clamp   old rule, as label_weak.py (50, 100, 1000)
    --force      overwrite --out
"""

import glob
import io
import json
import math
import multiprocessing
import os
import sys
import time

import numpy as np
import chess
import chess.engine
import chess.pgn

from extract import PIECE_ID
from label_weak import (CLAMP, EVAL_RECORD, JUDGE_CLAMP, MATE_SCORE,
                        read_games, terminal_cp)

# One row per position, including each game's final position, so the old
# rule's "eval after the move" is the next row's score.
LABEL = np.dtype([
    ("pieces",       np.int8, (64,)),
    ("stm",          np.int8),
    ("castling",     np.int8),
    ("ep",           np.int8),
    ("best_from",    np.int8),    # Stockfish's move; -1 where the game is over
    ("best_to",      np.int8),
    ("best_promo",   np.int8),
    ("played_from",  np.int8),    # the net's move; -1 on the final position
    ("played_to",    np.int8),
    ("played_promo", np.int8),
    ("best_cp",      np.int16),   # mover's view; mates as MATE_SCORE - moves
    ("best_mate",    np.int16),   # moves to mate: + mover mates, - mated
    ("played_cp",    np.int16),   # the played move, same position and depth
    ("played_mate",  np.int16),
    ("net_cp",       np.int16),   # the net's eval of this position, mover's view
    ("has_net",      np.int8),
    ("terminal",     np.int8),    # 1: the game is over in this position
    ("game",         np.int32),   # game number within this file
    ("ply",          np.int16),
])

_engine = None


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


# ---------------------------------------------------------------- labelling


def start_engine(engine_path, hash_mb):
    global _engine
    _engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    # One thread each: independent positions parallelise far better across
    # processes than Stockfish's search does across threads.
    _engine.configure({"Threads": 1, "Hash": hash_mb})


def scored(score):
    """(centipawns with mates folded in, moves to mate or 0) from a Score."""
    mate = score.mate()
    return score.score(mate_score=MATE_SCORE), (mate or 0)


def fill_position(row, board):
    for square, piece in board.piece_map().items():
        row["pieces"][square] = PIECE_ID[(piece.piece_type, piece.color)]
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
    row["stm"] = 1 if board.turn == chess.WHITE else 0
    row["castling"] = castling
    row["ep"] = board.ep_square if board.ep_square is not None else -1


def put_move(row, prefix, move):
    if move is None:
        row[prefix + "_from"] = row[prefix + "_to"] = -1
        row[prefix + "_promo"] = -1
    else:
        row[prefix + "_from"] = move.from_square
        row[prefix + "_to"] = move.to_square
        row[prefix + "_promo"] = move.promotion or 0


def label_game(game, engine, depth):
    """One LABEL row per position of the game, in order."""
    board = game.board()
    moves = list(game.mainline())
    rows = np.zeros(len(moves) + 1, dtype=LABEL)
    limit = chess.engine.Limit(depth=depth)
    net = None                  # the net's eval of the position about to move

    for ply in range(len(moves) + 1):
        row = rows[ply]
        node = moves[ply] if ply < len(moves) else None
        mover = board.turn
        fill_position(row, board)
        row["ply"] = ply

        settled = terminal_cp(board)
        if settled is not None:
            row["terminal"] = 1
            row["best_cp"] = settled if mover == chess.WHITE else -settled
            put_move(row, "best", None)
            put_move(row, "played", None)
        else:
            info = engine.analyse(board, limit)
            pv = info.get("pv")
            best = pv[0] if pv else None
            row["best_cp"], row["best_mate"] = scored(info["score"].pov(mover))
            put_move(row, "best", best)
            played = node.move if node is not None else None
            put_move(row, "played", played)
            if played is not None:
                if played == best:
                    row["played_cp"], row["played_mate"] = (
                        row["best_cp"], row["best_mate"])
                else:
                    # The played move alone, same position, same depth: what
                    # it is worth by the very search that preferred `best`.
                    other = engine.analyse(board, limit, root_moves=[played])
                    row["played_cp"], row["played_mate"] = scored(
                        other["score"].pov(mover))

        if net is not None:
            row["net_cp"], row["has_net"] = net, 1
        if node is None:
            break
        board.push(node.move)
        # selfplay.py comments each move with the net's eval of the position
        # it leads to - the next row's position, from its mover's view.
        evaluation = node.eval()
        net = (evaluation.pov(board.turn).score(mate_score=MATE_SCORE)
               if evaluation is not None else None)
    return rows


def label_one(payload):
    index, pgn_text, depth = payload
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    rows = label_game(game, _engine, depth) if game is not None else \
        np.zeros(0, dtype=LABEL)
    rows["game"] = index
    return index, rows


def resume_point(out_path, meta):
    """Games already labelled in out_path, after dropping a torn last game."""
    if not os.path.exists(out_path):
        with open(out_path + ".json", "w") as handle:
            json.dump(meta, handle)
        return 0
    with open(out_path + ".json") as handle:
        if json.load(handle) != meta:
            sys.exit(f"{out_path} was labelled with different settings - "
                     f"delete it and its .json, or use the same options")
    size = os.path.getsize(out_path) // LABEL.itemsize * LABEL.itemsize
    if not size:
        return 0
    rows = np.fromfile(out_path, dtype=LABEL, count=size // LABEL.itemsize)
    last = int(rows["game"][-1])
    # The last game may have been cut off mid-write; redo it.
    keep = int(np.searchsorted(rows["game"], last, side="left"))
    with open(out_path, "r+b") as handle:
        handle.truncate(keep * LABEL.itemsize)
    return last


def label():
    pattern, out_path = arg("--pgn"), arg("--out")
    if not pattern or not out_path:
        sys.exit("label needs --pgn and --out")
    depth = arg("--depth", 18, int)
    workers = arg("--workers", 4, int)
    engine_path = arg("--engine", "stockfish")
    hash_mb = arg("--hash", 128, int)
    limit = arg("--games", 0, int)
    part, _, parts = arg("--shard", "0/1").partition("/")
    shard = (int(part), int(parts or 1))
    if not 0 <= shard[0] < shard[1]:
        sys.exit(f"--shard {shard[0]}/{shard[1]} is not a valid i/n")

    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"no PGN files match {pattern!r} (quote the glob)")
    meta = {"pgn": pattern, "files": paths, "depth": depth,
            "shard": list(shard), "games": limit}
    done = resume_point(out_path, meta)

    print(f"reading {len(paths)} file(s) matching {pattern!r}, shard "
          f"{shard[0]} of {shard[1]}")
    print(f"Stockfish depth {depth}  |  {workers} workers  |  {hash_mb}MB "
          f"hash each  |  writing {out_path}")
    if done:
        print(f"resuming after {done:,} games already labelled")
    print(flush=True)

    work = ((i, text, depth) for i, text in
            enumerate(read_games(paths, limit, shard)) if i >= done)
    start = time.time()
    games = positions = moved = mistakes50 = mistakes20 = 0
    with open(out_path, "ab") as out, \
            multiprocessing.Pool(workers, start_engine,
                                 (engine_path, hash_mb)) as pool:
        # In order, so everything before the last written game is complete
        # and a restart can simply skip that many.
        for _, rows in pool.imap(label_one, work):
            out.write(rows.tobytes())
            out.flush()
            games += 1
            positions += len(rows)
            live = rows["played_from"] >= 0
            loss = (rows["best_cp"][live].astype(np.int32)
                    - rows["played_cp"][live])
            moved += int(live.sum())
            mistakes50 += int((loss > 50).sum())
            mistakes20 += int((loss > 20).sum())
            if games % 20 == 0:
                rate = positions / (time.time() - start)
                print(f"{done + games:>6} games | {positions:>9,} positions "
                      f"| {rate:5.1f} pos/s | mistakes per 100 moves: "
                      f"{mistakes20 / max(moved, 1) * 100:4.1f} over 20cp, "
                      f"{mistakes50 / max(moved, 1) * 100:4.1f} over 50cp",
                      flush=True)

    print(f"\nDone in {(time.time() - start) / 60:.1f} min: {games:,} games, "
          f"{positions:,} positions this run, "
          f"{os.path.getsize(out_path) // LABEL.itemsize:,} in the file")


# ---------------------------------------------------------------- selecting


def load_labels(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"no label files match {pattern!r} (quote the glob)")
    parts = []
    for i, path in enumerate(paths):
        count = os.path.getsize(path) // LABEL.itemsize
        rows = np.fromfile(path, dtype=LABEL, count=count)
        parts.append((i, rows))
    return paths, parts


def judge(rows, settings):
    """Per-row flags for one file's rows. Each returns a boolean array plus
    the size of each flag, so the report can show the spread."""
    n = len(rows)
    best = rows["best_cp"].astype(np.int32)
    has_best = rows["best_from"] >= 0
    same_game_next = np.zeros(n, dtype=bool)
    same_game_next[:-1] = rows["game"][1:] == rows["game"][:-1]
    after = np.zeros(n, dtype=np.int32)          # mover's view, old rule
    after[:-1] = -best[1:]

    flags = {}
    if settings["rule"] == "old":
        c = settings["clamp"]
        drop = np.clip(best, -c, c) - np.clip(after, -c, c)
        flags["blunder"] = (same_game_next & has_best
                            & (drop > settings["drop"]), drop)
        gap = np.abs(np.clip(rows["net_cp"].astype(np.int32), -c, c)
                     - np.clip(best, -c, c))
        flags["misjudged"] = ((rows["has_net"] == 1) & has_best
                              & (gap > settings["disagree"]), gap)
        return flags

    live = has_best & (rows["played_from"] >= 0)
    bm = rows["best_mate"].astype(np.int32)
    pm = rows["played_mate"].astype(np.int32)
    loss = best - rows["played_cp"].astype(np.int32)
    mating = bm > 0
    slower = mating & ((pm <= 0) | (pm > bm + settings["mate_slack"]))
    flags["conversion"] = (live & slower, np.where(pm > 0, pm - bm, 99))
    # Outside forced mates: the played move's score against the best's.
    ordinary = live & ~mating & (bm == 0)
    flags["mistake"] = (ordinary & (loss > settings["mistake_cp"]), loss)
    value = np.tanh(best / 400.0)
    net_value = np.tanh(rows["net_cp"].astype(np.float64) / 400.0)
    gap = np.abs(net_value - value)
    flags["misjudged"] = ((rows["has_net"] == 1) & has_best
                          & (gap > settings["misjudged"]), gap)
    return flags


def to_records(rows):
    out = np.zeros(len(rows), dtype=EVAL_RECORD)
    for field in ("pieces", "stm", "castling", "ep"):
        out[field] = rows[field]
    out["from_sq"] = rows["best_from"]
    out["to_sq"] = rows["best_to"]
    out["promo"] = rows["best_promo"]
    out["cp"] = np.clip(rows["best_cp"], -CLAMP, CLAMP)
    return out


def select():
    pattern = arg("--labels")
    if not pattern:
        sys.exit("select needs --labels")
    out_path = arg("--out")
    if out_path and os.path.exists(out_path) and "--force" not in sys.argv:
        sys.exit(f"{out_path} already exists. Pick another name, or --force.")
    settings = {
        "rule": arg("--rule", "new"),
        "mistake_cp": arg("--mistake-cp", 20, int),
        "mate_slack": arg("--mate-slack", 0, int),
        "misjudged": arg("--misjudged", 0.25, float),
        "drop": arg("--drop", 50, int),
        "disagree": arg("--disagree", 100, int),
        "clamp": arg("--clamp", JUDGE_CLAMP, int),
    }
    if settings["rule"] not in ("new", "old"):
        sys.exit("--rule is new or old")

    paths, parts = load_labels(pattern)
    positions = sum(len(rows) for _, rows in parts)
    games = sum(len(np.unique(rows["game"])) for _, rows in parts)
    print(f"{len(paths)} label file(s): {games:,} games, {positions:,} "
          f"positions")
    if settings["rule"] == "old":
        print(f"rule: old (label_weak.py) - drop over {settings['drop']}cp, "
              f"disagreement over {settings['disagree']}cp, judged within "
              f"+/-{settings['clamp']}cp")
    else:
        print(f"rule: new - mistakes over {settings['mistake_cp']}cp by the "
              f"same search, slower mates (slack {settings['mate_slack']}), "
              f"misjudged over {settings['misjudged']} on the -1..+1 scale")

    # The quality measure, comparable across rounds whatever the rule: moves
    # losing over 50cp, by label_weak's before/after test and by the
    # same-search test.
    moved = old50 = new50 = new20 = 0
    for _, rows in parts:
        old = judge(rows, dict(settings, rule="old", drop=50, clamp=JUDGE_CLAMP))
        new = judge(rows, dict(settings, rule="new", mistake_cp=50))
        new_20 = judge(rows, dict(settings, rule="new", mistake_cp=20))
        moved += int((rows["played_from"] >= 0).sum())
        old50 += int(old["blunder"][0].sum())
        new50 += int(new["mistake"][0].sum() + new["conversion"][0].sum())
        new20 += int(new_20["mistake"][0].sum()
                     + new_20["conversion"][0].sum())
    per100 = lambda k: k / max(moved, 1) * 100
    print(f"\nquality, per 100 moves played ({moved:,} moves):")
    print(f"  blunders over 50cp, old before/after test : {per100(old50):5.2f}")
    print(f"  over 50cp, same-search test (incl. mates) : {per100(new50):5.2f}")
    print(f"  over 20cp, same-search test (incl. mates) : {per100(new20):5.2f}")

    chosen, counts = [], {}
    for _, rows in parts:
        flags = judge(rows, settings)
        hit = np.zeros(len(rows), dtype=bool)
        for reason, (mask, _) in flags.items():
            counts[reason] = counts.get(reason, 0) + int(mask.sum())
            hit |= mask
        chosen.append(to_records(rows[hit]))
    records = np.concatenate(chosen) if chosen else \
        np.zeros(0, dtype=EVAL_RECORD)

    # The same position reached in two games is one record.
    as_bytes = records.view(np.dtype((np.void, EVAL_RECORD.itemsize)))
    _, first = np.unique(as_bytes, return_index=True)
    records = records[np.sort(first)]

    print("\nflagged:")
    for reason, count in sorted(counts.items()):
        print(f"  {reason:<12} {count:>10,}  ({count / positions:6.1%} of "
              f"positions)")
    cp = records["cp"].astype(np.int32)
    men = (records["pieces"] != 0).sum(axis=1)
    print(f"  records after merging overlaps and duplicates: {len(records):,} "
          f"({len(records) / max(positions, 1):.1%} of positions)")
    if len(records):
        print(f"  decisive (|eval| over 500cp): {(np.abs(cp) > 500).mean():.0%}"
              f"  |  endgames (12 or fewer pieces): {(men <= 12).mean():.0%}")

    if not out_path:
        print("\nnothing written - add --out to save the selection")
        return
    records.tofile(out_path)
    print(f"\nwrote {len(records):,} records "
          f"({len(records) * EVAL_RECORD.itemsize / 1e6:.0f} MB) to {out_path}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    if mode == "label":
        label()
    elif mode == "select":
        select()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
