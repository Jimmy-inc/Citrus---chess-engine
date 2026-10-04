"""
label_weak.py - find the net's mistakes in self-play games and turn them
into training records Stockfish has labelled.

Usage:
    python label_weak.py --pgn data/selfplay.pgn --out data/weakness.bin
    python label_weak.py --pgn 'data/selfplay*.pgn' --out data/weakness.bin
    python label_weak.py --pgn games.pgn --out w.bin --depth 22 --workers 24

Reads games written by selfplay.py, analyses every position once with
Stockfish, and writes the positions the net got wrong in the same 72-byte
format extract_evals.py produces - so train_v2.py can fine-tune on the
result with no new training code:

    python train_v2.py --data data/weakness.bin --init checkpoints/v2_best.pt

Two different mistakes are caught, because they fail in different ways.

  blunder    Stockfish's evaluation fell by more than --drop after the move
             the net chose. This is the obvious one, and mostly finds
             tactical slips.

  misjudged  the net's own evaluation of a position disagrees with
             Stockfish's by more than --disagree. This is the one that
             matters for the conversion weakness: shuffling into a
             repetition loses almost nothing on any single move, so a
             drop-based filter never sees it, but the value head being
             wrong about the position shows up here immediately.

Either way the record stored is the same shape: the position, Stockfish's
best move in it as the policy target, and Stockfish's score as the value
target. A blunder stores the position *before* the bad move, so the net
learns what it should have played. A misjudgement stores the position the
net was wrong about.

The net's evaluations come from the [%eval ...] comments selfplay.py
writes. A PGN without them still works, but only the blunder half runs -
the script says so rather than quietly finding half as much.

Options:
    --pgn        games to read; quote globs                     (required)
    --out        binary records to write             (default weakness.bin)
    --depth      Stockfish depth a position                    (default 18)
    --drop       centipawns lost to count as a blunder         (default 50)
    --disagree   net-vs-Stockfish gap to count as misjudged   (default 100)
    --clamp      evaluations are capped here before a move is judged, so
                 that +2500 sliding to +1200 is not read as a 1300cp
                 blunder when both are trivially winning. Raise it to catch
                 swings inside already-decided positions too; --clamp 3000
                 judges everything at face value          (default 1000)
    --workers    parallel Stockfish processes                   (default 4)
    --engine     Stockfish binary                       (default stockfish)
    --hash       hash MB per Stockfish process                (default 128)
    --games      stop after this many games                   (default all)
    --shard      i/n, to split one corpus across n jobs: each takes every
                 nth game. A coral node only has 8 cores, so sharding over
                 several nodes is the only way to use more than 8
                                                            (default 0/1)
    --survey     write nothing; report how many positions each threshold
                 would flag, so --drop and --disagree can be set from the
                 data rather than guessed at

On wolf, submit this rather than running it on the login node - it will
happily saturate every core it is given:

    sbatch -p general --time=02:00:00 -o ~/chessbot/logs/label.out \\
      --wrap "cd ~/chessbot && ./.venv/bin/python -u label_weak.py \\
              --pgn 'data/selfplay*.pgn' --out data/weakness.bin --workers 24"
"""

import glob
import io
import multiprocessing
import sys
import time

import numpy as np
import chess
import chess.engine
import chess.pgn

from extract import PIECE_ID

# ---------------------------------------------------------------- settings

MATE_SCORE = 3000   # centipawns for a forced mate; matches extract_evals.py
CLAMP = 3000        # stored scores are capped here; matches extract_evals.py

# Default for --clamp. Scores are capped here before a move is judged: the
# gap between +2500 and +1200 is not a mistake worth training on - both are
# winning - and uncapped, every trade in a won game reads as a big blunder.
JUDGE_CLAMP = 1000

# Must match EVAL_RECORD in extract_evals.py, and so train_v2.py.
EVAL_RECORD = np.dtype([
    ("pieces",   np.int8, (64,)),
    ("stm",      np.int8),
    ("castling", np.int8),
    ("ep",       np.int8),
    ("from_sq",  np.int8),
    ("to_sq",    np.int8),
    ("promo",    np.int8),
    ("cp",       np.int16),   # centipawns, from the side to move's view
])

_engine = None


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


# ---------------------------------------------------------------- scoring


def terminal_cp(board):
    """White's view of a finished game, or None while it is still going."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0
    return MATE_SCORE if outcome.winner == chess.WHITE else -MATE_SCORE


def look(engine, board, depth):
    """(centipawns from White's view, Stockfish's best move) for a position.

    A finished game is scored directly - Stockfish cannot analyse a position
    with no moves in it, and a blunder straight into mate is exactly the
    kind we most want to catch.
    """
    settled = terminal_cp(board)
    if settled is not None:
        return settled, None
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    pv = info.get("pv")
    return info["score"].white().score(mate_score=MATE_SCORE), \
        (pv[0] if pv else None)


def clamped(cp, limit):
    return max(-limit, min(limit, cp))


def thrown_away(before, after, mover, limit):
    """Centipawns the mover gave up, from their own point of view."""
    before, after = clamped(before, limit), clamped(after, limit)
    return (before - after) if mover == chess.WHITE else (after - before)


# ---------------------------------------------------------------- scanning


def scan(game, engine, settings):
    """Walk one game, returning a flag per position the net got wrong.

    Each flag is (fen, best move uci, centipawns from White's view, reason,
    how badly). Positions are returned rather than packed records so the
    parent can drop duplicates across the whole run before writing.
    """
    board = game.board()
    cp_before, best_before = look(engine, board, settings["depth"])
    flags = []
    analysed = 1
    had_net_eval = False

    for node in game.mainline():
        mover = board.turn
        before_fen = board.fen()

        board.push(node.move)
        cp_after, best_after = look(engine, board, settings["depth"])
        analysed += 1

        lost = thrown_away(cp_before, cp_after, mover, settings["clamp"])
        if lost > settings["drop"] and best_before is not None:
            flags.append((before_fen, best_before.uci(), cp_before,
                          "blunder", lost))

        # The net's own read of the position it just moved into. Same
        # position Stockfish just scored, so the two are directly comparable.
        net = node.eval()
        if net is not None:
            had_net_eval = True
            gap = abs(clamped(net.white().score(mate_score=MATE_SCORE),
                              settings["clamp"])
                      - clamped(cp_after, settings["clamp"]))
            if gap > settings["disagree"] and best_after is not None:
                flags.append((board.fen(), best_after.uci(), cp_after,
                              "misjudged", gap))

        cp_before, best_before = cp_after, best_after

    return flags, analysed, had_net_eval


def start_engine(engine_path, hash_mb):
    global _engine
    try:
        _engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    except FileNotFoundError:
        print(f"Stockfish not found at '{engine_path}'. Install it with: "
              f"brew install stockfish, or pass --engine /path/to/stockfish")
        raise
    # One thread each: many independent positions parallelise far better
    # across processes than Stockfish's own search does across threads.
    _engine.configure({"Threads": 1, "Hash": hash_mb})


def scan_one(payload):
    pgn_text, settings = payload
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return [], 0, False
    return scan(game, _engine, settings)


# ---------------------------------------------------------------- packing


def pack(buf, i, board, move, cp_white):
    """One record, laid out exactly as extract_evals.fill does."""
    pieces = buf["pieces"][i]
    pieces[:] = 0
    for square, piece in board.piece_map().items():
        pieces[square] = PIECE_ID[(piece.piece_type, piece.color)]

    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8

    # The value head is trained from the side to move's view, so store it
    # that way here too.
    cp = cp_white if board.turn == chess.WHITE else -cp_white
    cp = max(-CLAMP, min(CLAMP, int(cp)))

    buf["stm"][i] = 1 if board.turn == chess.WHITE else 0
    buf["castling"][i] = castling
    buf["ep"][i] = board.ep_square if board.ep_square is not None else -1
    buf["from_sq"][i] = move.from_square
    buf["to_sq"][i] = move.to_square
    buf["promo"][i] = move.promotion if move.promotion else 0
    buf["cp"][i] = cp


def report_survey(sizes, analysed):
    """What each threshold would cost and catch, before committing to one."""
    print(f"\n{analysed:,} positions probed\n")
    for reason, ladder in (("blunder", (25, 50, 100, 200, 400)),
                           ("misjudged", (100, 200, 300, 500, 800))):
        values = np.array(sizes.get(reason, []), dtype=np.float32)
        print(f"  {reason}")
        if not len(values):
            print("    none found\n")
            continue
        for cut in ladder:
            hits = int((values > cut).sum())
            print(f"    over {cut:>4}cp : {hits:>7,} positions  "
                  f"({hits / analysed * 100:5.2f}% of all)")
        print(f"    median {np.median(values):+.0f}cp, "
              f"90th percentile {np.percentile(values, 90):+.0f}cp\n")


def read_games(paths, limit, shard=(0, 1)):
    """PGN text a game at a time, so workers get something picklable.

    With --shard i/n each job takes every nth game. Every shard still
    parses the whole corpus, but parsing a game costs milliseconds against
    a second or more of Stockfish, so the waste is not worth avoiding.
    """
    part, parts = shard
    seen = taken = 0
    for path in paths:
        with open(path) as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                if seen % parts == part:
                    yield str(game)
                    taken += 1
                    if limit and taken >= limit:
                        return
                seen += 1


def main():
    pattern = arg("--pgn")
    if not pattern:
        print(__doc__)
        sys.exit(1)

    settings = {
        "depth": arg("--depth", 18, int),
        "drop": arg("--drop", 50, int),
        "disagree": arg("--disagree", 100, int),
        "clamp": arg("--clamp", JUDGE_CLAMP, int),
    }
    out_path = arg("--out", "weakness.bin")
    workers = arg("--workers", 4, int)
    engine_path = arg("--engine", "stockfish")
    hash_mb = arg("--hash", 128, int)
    limit = arg("--games", 0, int)
    part, _, parts = arg("--shard", "0/1").partition("/")
    shard = (int(part), int(parts or 1))
    if not 0 <= shard[0] < shard[1]:
        print(f"--shard {shard[0]}/{shard[1]} is not a valid i/n")
        sys.exit(1)
    survey = "--survey" in sys.argv

    # A survey keeps everything and sorts out the thresholds afterwards,
    # which is the whole point - you cannot see what a cutoff would have
    # caught if the cutoff already threw it away.
    if survey:
        settings["drop"] = -10 ** 6
        settings["disagree"] = -1

    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no PGN files match {pattern!r} "
              f"(quote the glob or the shell eats it)")
        sys.exit(1)

    print(f"reading {len(paths)} file(s) matching {pattern!r}")
    print(f"Stockfish depth {settings['depth']}  |  {workers} workers  |  "
          f"{hash_mb}MB hash each"
          + (f"  |  shard {shard[0]} of {shard[1]}" if shard[1] > 1 else ""))
    if survey:
        print("survey only - measuring the spread, writing no records\n")
    else:
        print(f"flagging drops over {settings['drop']}cp and disagreements "
              f"over {settings['disagree']}cp, judged within "
              f"+/-{settings['clamp']}cp")
        print(f"writing {out_path}\n")

    games = 0
    analysed = 0
    reasons = {}
    sizes = {}
    seen = set()
    flagged = []
    net_evals_seen = False
    start = time.time()

    work = ((text, settings) for text in read_games(paths, limit, shard))
    with multiprocessing.Pool(workers, start_engine,
                              (engine_path, hash_mb)) as pool:
        for flags, positions, had_net in pool.imap_unordered(scan_one, work):
            games += 1
            analysed += positions
            net_evals_seen = net_evals_seen or had_net

            for fen, uci, cp, reason, size in flags:
                # Every observation counts towards the spread, but only the
                # first sighting of a position becomes a record: a position
                # is both the one a move led into and the one the next move
                # starts from, so the two criteria collide constantly.
                sizes.setdefault(reason, []).append(size)
                if fen in seen:
                    continue
                seen.add(fen)
                flagged.append((fen, uci, cp))
                reasons[reason] = reasons.get(reason, 0) + 1

            if games % 20 == 0:
                rate = analysed / (time.time() - start)
                print(f"{games:>6} games | {analysed:>8,} positions | "
                      f"{len(flagged):>7,} flagged | {rate:5.1f} pos/s")

    if not net_evals_seen:
        print("\nwarning: no [%eval] comments in these games, so only "
              "blunders were found. Games from selfplay.py carry them; "
              "games from elsewhere do not.")

    if survey:
        report_survey(sizes, analysed)
        print(f"scanned {games:,} games in {(time.time()-start)/60:.1f} min. "
              f"Pick thresholds, then run again without --survey.")
        return

    buf = np.zeros(max(len(flagged), 1), dtype=EVAL_RECORD)
    written = skipped = 0
    for fen, uci, cp in flagged:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            skipped += 1
            continue
        pack(buf, written, board, move, cp)
        written += 1

    with open(out_path, "wb") as out:
        out.write(buf[:written].tobytes())

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  games scanned    : {games:,}")
    print(f"  positions probed : {analysed:,}")
    for reason, count in sorted(reasons.items()):
        print(f"  {reason:<16} : {count:,}")
    if skipped:
        print(f"  illegal, skipped : {skipped:,}")
    print(f"  records written  : {written:,}")
    print(f"  file size        : {written * EVAL_RECORD.itemsize / 1e6:.1f} MB")
    if analysed:
        print(f"  flagged          : {len(flagged)/analysed*100:.1f}% "
              f"of positions")

    if written:
        data = np.memmap(out_path, dtype=EVAL_RECORD, mode="r")
        cp = data["cp"].astype(np.float32)
        print(f"\n  score spread: mean {cp.mean():+.0f}cp, "
              f"median {np.median(cp):+.0f}cp")
        print(f"  {(np.abs(cp) < 50).mean()*100:.0f}% near-equal, "
              f"{(np.abs(cp) > 500).mean()*100:.0f}% decisive")
        print(f"\nFine-tune on it:  python train_v2.py --data {out_path} "
              f"--init checkpoints/v2_best.pt")


if __name__ == "__main__":
    main()
