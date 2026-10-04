"""
mine_hard.py - find the positions in an eval file that the net gets wrong,
without self-play, and with Stockfish only where it is needed.

Every record in evals_d22.bin already carries a depth-22 Stockfish score and
best move, so there are two kinds of mistake to look for:

  * misjudged - the value head's opinion of the position is far from
    Stockfish's. One forward pass per record finds these.
  * mistakes  - the bot, searching as it would in a game, plays a move
    Stockfish rates worse than its own. Most positions where the two
    differ are not mistakes at all: chess often has two or three equally
    good moves, and Stockfish's pick among them is a coin flip. So when
    the bot's move differs, Stockfish scores both moves at the same depth,
    and only a real loss counts.

Searching is slow (about 0.4s a position), so it cannot cover 200M records.
It runs on the positions whose Stockfish move surprised the net's policy
most, which the scoring pass already measured - that is where a search is
most likely to go wrong.

The training record for a flagged position is the original one: Stockfish's
best move and evaluation, which is exactly what the net should learn there.

Three steps, each saving results so a killed job resumes where it stopped:

    # 1. GPU: score every record (5 bytes each, ~1GB for evals_d22.bin)
    python mine_hard.py score --ckpt checkpoints/mix_best.pt --shard 0/2

    # 2. GPU + a few CPUs: search the most surprising positions, and have
    #    Stockfish judge the ones where the bot's move differs
    python mine_hard.py mistakes --ckpt checkpoints/mix_best.pt \\
        --candidates 200000 --shard 0/2

    # 3. anywhere: report, then write a selection. Thresholds apply here,
    #    so they can be changed without searching again
    python mine_hard.py select --ckpt checkpoints/mix_best.pt --mistake-cp 20
    python mine_hard.py select --ckpt checkpoints/mix_best.pt --mistake-cp 20 \\
        --count 200000 --out data/hard_mix.bin

Options:
    --ckpt          net to find the weaknesses of                 (required)
    --data          eval records              (default data/evals_d22.bin)
    --scores-dir    where result files go              (default data/scores)
    --shard i/n     do only the i-th of n equal parts          (default 0/1)

  score
    --limit N       score only the first N records - for timing tests
    --batch         positions per forward pass                (default 2048)
    --include-val   also score train_v2.py's held-out validation tail

  mistakes
    --candidates N  how many of the most surprising positions to search,
                    across all shards                       (default 200000)
    --sims          simulations per search                     (default 800)
    --depth         Stockfish depth for judging both moves      (default 16)
    --engine        Stockfish binary                    (default stockfish)
    --threads       Stockfish threads                            (default 3)
    --hash          Stockfish hash in MB                       (default 256)

  select
    --mistake-cp T  keep positions where the bot's move lost more than T
                    centipawns                              (default: none)
    --count N       also keep the N worst value errors           (default 0)
    --policy-count N  also keep the N biggest policy surprises, searched or
                    not                                          (default 0)
    --out           write the selection; without it, only report
    --seed          shuffle seed for the output                  (default 0)
    --force         overwrite --out
"""

import glob
import json
import math
import os
import sys
import time

import chess
import numpy as np
import torch
import torch.nn.functional as F

from encoding_v2 import MOVE_LOOKUP, canonicalize, planes_from_pieces
from train_v2 import (EVAL_RECORD, VAL_POSITIONS, amp_dtype, castle_onto_rook,
                      pick_device)

SCALE = 400                       # cp per unit of tanh, as in train_v2.py
MATE_CP = 3000                    # as extract_evals.py scores a forced mate
SCORE = np.dtype([("err", np.float16),     # net value minus Stockfish's
                  ("nll", np.float16),     # -log p(Stockfish's move)
                  ("men", np.int8)])       # pieces on the board, kings too
JUDGED = np.dtype([("index", np.int64),    # record number in --data
                   ("verdict", np.int8),   # see the constants below
                   ("best_cp", np.int16),  # Stockfish's move, mover's view
                   ("bot_cp", np.int16),   # the bot's move, same depth
                   ("bot_move", np.uint16)])
TODO, SKIPPED, SAME, DIFFERENT = -2, -1, 0, 1
REPORT_EVERY = 2_000_000
# Stockfish at depth 16 answers in seconds. One that has said nothing for
# this long is hung, and is killed so the position can be retried.
STOCKFISH_PATIENCE = 600
# No position finished for this long means something below the script is
# stuck (twice, Oct 2, both shards froze together for 35 hours). Better to
# say where and exit, so a queued resubmit can carry on, than to idle.
STALL_LIMIT = 30 * 60


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def settings():
    ckpt_path = arg("--ckpt")
    data_path = arg("--data", "data/evals_d22.bin")
    folder = arg("--scores-dir", "data/scores")
    prefix = os.path.join(folder, f"{stem(ckpt_path)}_on_{stem(data_path)}")
    shard, shards = (int(p) for p in arg("--shard", "0/1").split("/"))
    if not 0 <= shard < shards:
        sys.exit(f"--shard {shard}/{shards}: the first number counts from 0")
    return ckpt_path, data_path, prefix, shard, shards


def open_results(path, meta, dtype, length, blank):
    """
    A results file for this shard, created or reopened for resuming.

    Rows still holding `blank` are the ones left to do. The settings are
    kept beside it, so resuming with different ones is refused rather than
    mixing two kinds of result in one file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        with open(path + ".json") as handle:
            if json.load(handle) != meta:
                sys.exit(f"{path} was made with different settings - delete "
                         f"it and its .json to start over, or pass the same "
                         f"options as before")
        return np.load(path, mmap_mode="r+")
    results = np.lib.format.open_memmap(path, mode="w+", dtype=dtype,
                                        shape=(length,))
    blank(results)
    results.flush()
    with open(path + ".json", "w") as handle:
        json.dump(meta, handle)
    return results


def load_net(ckpt_path):
    # Imported here so `select` runs without the search's dependencies.
    from search_v2 import load
    device = pick_device()
    model, ckpt = load(ckpt_path)
    model.to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"{ckpt_path} (step {ckpt.get('step', 0):,}) on {device}")
    return model, device


# ---------------------------------------------------------------- score


def prepare(recs):
    """Network input and targets for a slice of records."""
    pieces, castling, ep, from_sq, to_sq = canonicalize(
        recs["pieces"], recs["stm"], recs["castling"], recs["ep"],
        recs["from_sq"], recs["to_sq"])
    to_sq = castle_onto_rook(pieces, from_sq, to_sq)
    x = planes_from_pieces(pieces, castling, ep, np.uint8)
    move = MOVE_LOOKUP[from_sq.astype(np.int64), to_sq.astype(np.int64),
                       recs["promo"].astype(np.int64)].astype(np.int64)
    value = np.tanh(recs["cp"].astype(np.float32) / SCALE)
    men = (pieces != 0).sum(axis=1).astype(np.int8)
    return x, move, value, men


def score():
    ckpt_path, data_path, prefix, shard, shards = settings()
    batch = arg("--batch", 2048, int)

    data = np.memmap(data_path, dtype=EVAL_RECORD, mode="r")
    region = len(data)
    if "--include-val" not in sys.argv:
        region -= min(VAL_POSITIONS, len(data) // 100)
    region = min(region, arg("--limit", region, int))
    start = region * shard // shards
    stop = region * (shard + 1) // shards
    length = stop - start

    path = f"{prefix}_{shard}of{shards}.npy"
    meta = {"ckpt": ckpt_path, "data": data_path, "start": start, "stop": stop}

    def blank(results):
        results["err"] = np.nan          # NaN marks "not scored yet"

    scores = open_results(path, meta, SCORE, length, blank)
    unscored = np.flatnonzero(np.isnan(scores["err"]))
    done = int(unscored[0]) if len(unscored) else length

    model, device = load_net(ckpt_path)
    print(f"scoring records {start:,} to {stop:,} of {data_path} "
          f"(shard {shard}/{shards}), {length:,} in all")
    if done:
        print(f"resuming: {done:,} already scored")
    print(f"writing {path}\n", flush=True)

    def launch(pos):
        recs = np.array(data[start + pos:start + min(pos + batch, length)])
        x, move, value, men = prepare(recs)
        x = torch.from_numpy(x).to(device, non_blocking=True).float()
        with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype(),
                                             enabled=device.type == "cuda"):
            logits, predicted = model(x)
        # Moves the move table cannot express would give nonsense; score
        # them as fully expected so they are never picked as surprises.
        valid = torch.from_numpy(move >= 0).to(device)
        target = torch.from_numpy(np.maximum(move, 0)).to(device)
        nll = F.cross_entropy(logits.float(), target, reduction="none")
        nll = torch.where(valid, nll, torch.zeros_like(nll))
        err = predicted.float() - torch.from_numpy(value).to(device)
        return pos, torch.stack([err, nll], dim=1), men

    began = time.time()
    first = done
    next_report = done + REPORT_EVERY
    pending = launch(done) if done < length else None
    while pending:
        pos, result, men = pending
        after = pos + batch
        # The GPU works on this batch while the CPU prepares the next one.
        pending = launch(after) if after < length else None
        result = result.cpu().numpy()
        scores["err"][pos:pos + len(men)] = result[:, 0]
        scores["nll"][pos:pos + len(men)] = result[:, 1]
        scores["men"][pos:pos + len(men)] = men

        if after >= next_report or not pending:
            scores.flush()
            finished = min(after, length)
            rate = (finished - first) / (time.time() - began)
            left = (length - finished) / rate
            print(f"  {finished:>12,} / {length:,}  "
                  f"{finished / length:6.1%}  |  {rate:,.0f} pos/s  |"
                  f"  {left / 3600:.1f}h left", flush=True)
            next_report += REPORT_EVERY

    scores.flush()
    print("\ndone. Once every shard has finished, run `mistakes` or `select`.")


def load_scores(ckpt_path, data_path, prefix, quiet=False):
    """Every shard's scores, the record each belongs to, and completeness."""
    paths = sorted(glob.glob(prefix + "_[0-9]*of[0-9]*.npy"))
    if not paths:
        sys.exit(f"no scores for {ckpt_path} on {data_path} - run score first")
    splits = {p.rsplit("of", 1)[1] for p in paths}
    if len(splits) > 1:
        sys.exit(f"score files from different --shard splits: {paths}")

    parts, index = [], []
    complete = True
    for path in paths:
        with open(path + ".json") as handle:
            meta = json.load(handle)
        s = np.load(path, mmap_mode="r")
        scored = ~np.isnan(s["err"])
        complete &= bool(scored.all())
        if not quiet:
            print(f"  {os.path.basename(path)}  {scored.sum():>12,} / "
                  f"{len(s):,} scored")
        parts.append(np.array(s[scored]))
        index.append(np.flatnonzero(scored) + meta["start"])
    expected = int(paths[0].rsplit("of", 1)[1].split(".")[0])
    if len(paths) != expected:
        complete = False
        if not quiet:
            print(f"  only {len(paths)} of {expected} shards present")
    return np.concatenate(parts), np.concatenate(index), complete


def worst(values, count):
    """Positions of the `count` largest values, unordered."""
    count = min(count, len(values))
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.argpartition(values, len(values) - count)[len(values) - count:]


# ---------------------------------------------------------------- mistakes


PIECES = [None] + [chess.Piece(kind, colour)
                   for colour in (True, False) for kind in range(1, 7)]


def plausible(pieces):
    """
    False for positions no game could reach, one row per record.

    Lichess's evaluation database includes positions people set up on the
    analysis board - five queens beside eight pawns, nine queens - which
    python-chess accepts but Stockfish can crash on. A side can only have
    more than one queen, or more than two of another piece, by promoting,
    and every promotion uses up a pawn.
    """
    ok = np.ones(len(pieces), dtype=bool)
    for base in (0, 6):
        pawns, knights, bishops, rooks, queens, kings = (
            (pieces == base + kind).sum(axis=1) for kind in range(1, 7))
        promoted = (np.maximum(knights - 2, 0) + np.maximum(bishops - 2, 0)
                    + np.maximum(rooks - 2, 0) + np.maximum(queens - 1, 0))
        ok &= (kings == 1) & (pawns + promoted <= 8)
    return ok


def top_plausible(values, index, count, data):
    """Record numbers of the `count` largest values among real positions."""
    take = count
    while count > 0:
        top = worst(values, min(int(take * 1.05) + 100, len(values)))
        rows = index[top[np.argsort(-values[top], kind="stable")]]
        order = np.argsort(rows)
        ok = np.empty(len(rows), dtype=bool)
        ok[order] = plausible(np.asarray(data[rows[order]]["pieces"]))
        if ok.sum() >= count or len(top) == len(values):
            return rows[ok][:count]
        take *= 2
    return np.zeros(0, dtype=np.int64)


def board_of(record):
    """A python-chess board for one record, as extract_evals.py wrote it."""
    board = chess.Board(None)
    for square in np.flatnonzero(record["pieces"]):
        board.set_piece_at(int(square), PIECES[int(record["pieces"][square])])
    board.turn = bool(record["stm"])
    rights = int(record["castling"])
    board.set_castling_fen("".join(
        c for bit, c in zip((1, 2, 4, 8), "KQkq") if rights & bit) or "-")
    board.ep_square = int(record["ep"]) if record["ep"] >= 0 else None
    return board


def pack(move):
    return (move.from_square | move.to_square << 6
            | (move.promotion or 0) << 12)


def unpack(packed):
    packed = int(packed)
    return chess.Move(packed & 63, (packed >> 6) & 63, (packed >> 12) or None)


def mistakes():
    import chess.engine
    import signal
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout
    import encoding_v2
    import search_v2 as engine_v2
    # The default now, but older copies of encoding_v2 had it off: positions
    # where only the old lookup kept the bot from castling are not mistakes.
    encoding_v2.CASTLE_ONTO_ROOK = True

    ckpt_path, data_path, prefix, shard, shards = settings()
    wanted = arg("--candidates", 200_000, int)
    sims = arg("--sims", 800, int)
    depth = arg("--depth", 16, int)

    scores, index, complete = load_scores(ckpt_path, data_path, prefix,
                                          quiet=True)
    if not complete:
        sys.exit("scoring has not finished on every shard yet - the "
                 "candidates would change once it has")

    # Most surprising first, so a run cut short has still done the best
    # part. Shards take every n-th, so each gets the same mix.
    nll = scores["nll"].astype(np.float32)
    top = worst(nll, wanted)
    ranked = index[top][np.lexsort((index[top], -nll[top]))]
    mine = ranked[shard::shards]
    lowest = math.exp(-nll[top].min()) if len(top) else 0

    path = f"{prefix}_mistakes_{shard}of{shards}.npy"
    meta = {"ckpt": ckpt_path, "data": data_path, "candidates": wanted,
            "sims": sims, "depth": depth}

    def blank(results):
        results["index"] = mine
        results["verdict"] = TODO

    judged = open_results(path, meta, JUDGED, len(mine), blank)
    todo = np.flatnonzero(judged["verdict"] == TODO)

    model, _ = load_net(ckpt_path)

    def start_stockfish():
        try:
            engine = chess.engine.SimpleEngine.popen_uci(
                arg("--engine", "stockfish"))
        except FileNotFoundError:
            sys.exit("Stockfish not found - pass --engine /path/to/stockfish")
        options = {"Threads": arg("--threads", 3, int),
                   "Hash": arg("--hash", 256, int)}
        # Stockfish pins its threads to CPUs it reads from the machine, but
        # a Slurm job may only use some of them, and a refused pin makes
        # Stockfish exit. A few threads gain nothing from pinning anyway.
        if "NumaPolicy" in engine.options:
            options["NumaPolicy"] = "none"
        engine.configure(options)
        return engine

    sf = [start_stockfish()]
    deaths = [0]                  # in a row, so a broken setup still stops
    limit = chess.engine.Limit(depth=depth)

    print(f"searching {len(mine):,} of the {wanted:,} most surprising "
          f"positions (shard {shard}/{shards}): Stockfish's move given "
          f"under {lowest:.2%} by the policy")
    print(f"{sims} sims a search; differing moves judged by Stockfish at "
          f"depth {depth}")
    if len(todo) < len(mine):
        print(f"resuming: {len(mine) - len(todo):,} already done")
    print(f"writing {path}\n", flush=True)

    def judge(row, board, best, played):
        """Stockfish's score for both moves, or None if it keeps dying.

        A Stockfish that dies on one odd position is restarted and the
        position retried once, then skipped. Three deaths in a row means
        something is wrong with Stockfish itself, and the job stops.
        """
        for attempt in (1, 2):
            try:
                cp = []
                for move in (best, played):
                    info = sf[0].analyse(board, limit, root_moves=[move])
                    cp.append(info["score"].pov(board.turn)
                              .score(mate_score=MATE_CP))
                deaths[0] = 0
                return row, cp
            except chess.engine.EngineTerminatedError as error:
                deaths[0] += 1
                print(f"  Stockfish died ({error}) on {board.fen()} "
                      f"moves {best.uci()} {played.uci()}", flush=True)
                if deaths[0] >= 3:
                    raise
                sf[0] = start_stockfish()
        return row, None

    def record(row, cp):
        if cp is None:
            judged["verdict"][row] = SKIPPED
            return
        judged["best_cp"][row], judged["bot_cp"][row] = cp
        judged["verdict"][row] = DIFFERENT

    data = np.memmap(data_path, dtype=EVAL_RECORD, mode="r")
    # Stockfish runs in its own process, so a thread waiting on it costs
    # the search nothing: it judges one position while the GPU searches
    # the next.
    pool = ThreadPoolExecutor(max_workers=1)
    waiting = []

    # What the loop is doing, and when it last finished a position, for
    # the watchdog to report if everything stops.
    stage = ["starting"]
    beat = [time.time()]

    def watchdog():
        warned = False
        while True:
            time.sleep(min(60, STALL_LIMIT / 6))
            quiet = time.time() - beat[0]
            if quiet > STALL_LIMIT:
                # Judgements already back from Stockfish are kept. Only
                # memory is touched here: if the file system is what hung,
                # a flush would hang the watchdog too.
                for future in list(waiting):
                    if future.done() and future.exception() is None:
                        record(*future.result())
                print(f"\n  no position finished for {quiet / 60:.0f} "
                      f"minutes, stuck {stage[0]}. Exiting so the job ends "
                      f"instead of idling - resubmit to carry on from here.",
                      flush=True)
                os._exit(3)
            if quiet > STALL_LIMIT / 3 and not warned:
                print(f"  (nothing finished for {quiet / 60:.0f} minutes; "
                      f"currently {stage[0]})", flush=True)
                warned = True
            elif quiet < STALL_LIMIT / 6:
                warned = False

    threading.Thread(target=watchdog, daemon=True).start()

    def collect(future):
        """A judgement, killing Stockfish if it stops answering. The kill
        makes judge() see a dead engine, restart it and retry."""
        stage[0] = "waiting for Stockfish"
        while True:
            try:
                return future.result(timeout=STOCKFISH_PATIENCE)
            except FutureTimeout:
                print(f"  Stockfish silent for {STOCKFISH_PATIENCE // 60} "
                      f"minutes - killing it", flush=True)
                try:
                    os.kill(sf[0].transport.get_pid(), signal.SIGKILL)
                except (OSError, AttributeError):
                    pass

    began = time.time()
    count = {SAME: 0, DIFFERENT: 0, SKIPPED: 0}
    stalled = 0.0
    for done, row in enumerate(todo, 1):
        rec = data[int(judged["index"][row])]
        board = board_of(rec)
        best = chess.Move(int(rec["from_sq"]), int(rec["to_sq"]),
                          int(rec["promo"]) or None)
        if (not plausible(rec["pieces"][None])[0]
                or not board.is_valid() or board.is_game_over()
                or best not in board.legal_moves):
            judged["verdict"][row] = SKIPPED
            count[SKIPPED] += 1
        else:
            # Lichess writes castling king-onto-rook (e1h1); the search plays
            # e1g1. Same move, so compare them in one spelling.
            best = board.parse_uci(board.uci(best))
            engine_v2.clear_cache()
            stage[0] = f"searching {board.fen()}"
            root = engine_v2.run_search(model, board, sims)
            played = engine_v2.choose(root)[0]
            judged["bot_move"][row] = pack(played)
            if played == best:
                judged["verdict"][row] = SAME
                count[SAME] += 1
            else:
                waiting.append(pool.submit(judge, row, board, best, played))
                count[DIFFERENT] += 1

        while waiting and (waiting[0].done() or len(waiting) > 32):
            # A full queue means the search is idle until Stockfish catches
            # up; the share of time spent here says which side is the limit.
            stall = time.time()
            record(*collect(waiting.pop(0)))
            stalled += time.time() - stall
        beat[0] = time.time()

        if done % 1000 == 0 or done == len(todo):
            stage[0] = "saving results"
            judged.flush()
            rate = done / (time.time() - began)
            searched = count[SAME] + count[DIFFERENT]
            finished = judged["verdict"] == DIFFERENT
            lost = (judged["best_cp"][finished].astype(np.int32)
                    - judged["bot_cp"][finished])
            print(f"  {done:>8,} / {len(todo):,}  |  {rate * 3600:,.0f}/h  |  "
                  f"{(len(todo) - done) / rate / 3600:.1f}h left  |  same "
                  f"move {count[SAME] / max(searched, 1):.0%}  |  worse by "
                  f">20cp so far: {np.sum(lost > 20):,}  |  waiting on "
                  f"Stockfish {stalled / (time.time() - began):.0%}",
                  flush=True)

    for future in waiting:
        record(*collect(future))
    judged.flush()
    pool.shutdown()
    sf[0].quit()
    print("\ndone. Once every shard has finished, run `select`.")


def load_mistakes(prefix):
    paths = sorted(glob.glob(prefix + "_mistakes_*of*.npy"))
    if not paths:
        return None
    return np.concatenate([np.array(np.load(p, mmap_mode="r")) for p in paths])


# ---------------------------------------------------------------- select


def as_value(cp):
    return math.tanh(cp / SCALE)


def select():
    ckpt_path, data_path, prefix, _, _ = settings()
    threshold = arg("--mistake-cp", None, int)
    count = arg("--count", 0, int)
    policy_count = arg("--policy-count", 0, int)
    out_path = arg("--out")

    if out_path and os.path.exists(out_path) and "--force" not in sys.argv:
        sys.exit(f"{out_path} already exists. Pick another name, or --force.")

    print(f"scores for {ckpt_path} on {data_path}:")
    scores, index, _ = load_scores(ckpt_path, data_path, prefix)
    err = np.abs(scores["err"].astype(np.float32))
    nll = scores["nll"].astype(np.float32)
    total = len(scores)

    print(f"\n{total:,} positions scored")
    print("how far the value head is from Stockfish (value units, 0.25 is "
          "about 100cp in a level position):")
    for q in (50, 90, 99, 99.9):
        print(f"  {q:>5}% of positions within {np.percentile(err, q):.3f}")
    for t in (0.25, 0.5, 0.75, 1.0):
        print(f"  error above {t:<4}  {np.sum(err > t):>12,}  "
              f"({np.mean(err > t):.2%})")
    print(f"Stockfish's move given under 1% by the policy: "
          f"{np.mean(nll > math.log(100)):.1%} of positions")

    data = np.memmap(data_path, dtype=EVAL_RECORD, mode="r")
    chosen = [top_plausible(err, index, count, data),
              top_plausible(nll, index, policy_count, data)]

    judged = load_mistakes(prefix)
    if judged is not None:
        done = judged[judged["verdict"] != TODO]
        searched = done[done["verdict"] >= SAME]
        differ = searched[searched["verdict"] == DIFFERENT]
        lost = differ["best_cp"].astype(np.int32) - differ["bot_cp"]
        print(f"\nsearched {len(searched):,} surprising positions "
              f"({len(judged) - len(done):,} still to do, "
              f"{len(done) - len(searched):,} impossible or unreadable, skipped):")
        print(f"  same move as Stockfish   {len(searched) - len(differ):>9,}")
        print(f"  different move           {len(differ):>9,}")
        for low, high, name in ((-10**6, 10, "as good (within 10cp)"),
                                (10, 20, "10-20cp worse"),
                                (20, 60, "20-60cp worse"),
                                (60, 150, "60-150cp worse"),
                                (150, 10**6, "over 150cp worse")):
            k = np.sum((lost > low) & (lost <= high))
            print(f"    {name:<24} {k:>9,}  "
                  f"({k / max(len(searched), 1):.1%} of searched)")
        if threshold is not None:
            chosen.append(differ["index"][lost > threshold])
    elif threshold is not None:
        sys.exit("--mistake-cp given, but no mistakes files - run "
                 "`mistakes` first")

    parts = [len(c) for c in chosen]
    records_at = np.unique(np.concatenate(chosen)).astype(np.int64)
    if not len(records_at):
        print("\nPass --mistake-cp, --count or --policy-count to see a "
              "selection.")
        return

    print(f"\nselection: {parts[0]:,} worst value errors + {parts[1]:,} "
          f"policy surprises", end="")
    if threshold is not None:
        print(f" + {parts[2]:,} mistakes over {threshold}cp", end="")
    print(f" = {len(records_at):,} after overlap")

    print(f"reading {len(records_at):,} records...", flush=True)
    recs = data[records_at]

    # What kinds of positions they are. The conversion weakness would show
    # up as endgames, and as "Stockfish says winning" positions.
    men = (recs["pieces"] != 0).sum(axis=1)
    print("\n  pieces   share of selection   share of all positions")
    for low, high in ((2, 6), (7, 12), (13, 20), (21, 32)):
        here = (men >= low) & (men <= high)
        everywhere = (scores["men"] >= low) & (scores["men"] <= high)
        print(f"  {low:>2}-{high:<2}    {here.mean():>17.1%}   "
              f"{everywhere.mean():>21.1%}")
    target = recs["cp"]
    print("\n  Stockfish says           share of selection")
    for name, low, high in (("losing  (< -300)", -10**6, -300),
                            ("worse   (-300..-100)", -300, -100),
                            ("level   (-100..100)", -100, 100),
                            ("better  (100..300)", 100, 300),
                            ("winning (> 300)", 300, 10**6)):
        here = (target >= low) & (target < high)
        print(f"  {name:<24} {here.mean():>7.1%}")

    if not out_path:
        print("\nnothing written - add --out to save the selection")
        return

    rng = np.random.default_rng(arg("--seed", 0, int))
    recs = recs[rng.permutation(len(recs))]
    recs.tofile(out_path)
    print(f"\nwrote {len(recs):,} records "
          f"({len(recs) * EVAL_RECORD.itemsize / 1e6:.0f} MB) to {out_path}")
    print("Mix it with ordinary data before training, as with the weakness "
          "set:\n  python mix_data.py --weak " + out_path + " --out ...")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    if mode not in ("score", "mistakes", "select") or not arg("--ckpt"):
        print(__doc__)
        sys.exit(1)
    {"score": score, "mistakes": mistakes, "select": select}[mode]()


if __name__ == "__main__":
    main()
