"""
extract_evals.py - turn the Lichess Stockfish evaluation database into
binary records with continuous value targets.

Usage:
    python extract_evals.py data/lichess_db_eval.jsonl.zst data/evals.bin
    python extract_evals.py data/lichess_db_eval.jsonl.zst data/evals.bin 10000000
    python extract_evals.py IN OUT --min-depth 14
    python extract_evals.py IN OUT --min-depth 25 --max 50000000

The third positional argument (or --max) caps how many records to write.
The file is enormous and zstd decompresses as a stream, so capping lets you
stop early rather than processing all of it.

--min-depth sets how deep an evaluation must be to be kept. Lower keeps more
positions with noisier labels; higher keeps fewer, cleaner ones. The default
of 18 discards under 10% of the database.

Records are 72 bytes and are NOT the same format as extract.py produces -
they carry a centipawn score instead of a game result, so they cannot be
concatenated with your existing .bin files.
"""

import json
import sys
import time

import numpy as np
import chess
import zstandard

from extract import PIECE_ID

# ---------------------------------------------------------------- settings

MIN_DEPTH = 18          # skip shallow browser evaluations; see --min-depth
MATE_SCORE = 3000       # centipawn value assigned to a forced mate
CLAMP = 3000            # scores beyond this are capped
BUFFER_SIZE = 100_000
PERSPECTIVE_SAMPLE = 20_000   # positions used to work out the sign convention

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

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def best_pv(entry, min_depth=MIN_DEPTH):
    """Pick the deepest evaluation, and its first principal variation."""
    best = None
    for ev in entry.get("evals", ()):
        if ev.get("depth", 0) < min_depth:
            continue
        if not ev.get("pvs"):
            continue
        if best is None or ev["depth"] > best["depth"]:
            best = ev
    if best is None:
        return None, None
    return best, best["pvs"][0]


def score_of(pv):
    """Centipawns, with mates mapped onto a large finite score."""
    if pv.get("mate") is not None:
        mate = pv["mate"]
        if mate == 0:
            return None
        return MATE_SCORE if mate > 0 else -MATE_SCORE
    cp = pv.get("cp")
    if cp is None:
        return None
    return max(-CLAMP, min(CLAMP, int(cp)))


def material_balance(board):
    """Positive when White has more material."""
    total = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUE[piece.piece_type]
        total += value if piece.color == chess.WHITE else -value
    return total


def board_from_fen(fen):
    """The database FEN omits the move counters; python-chess wants them."""
    parts = fen.split()
    while len(parts) < 6:
        parts.append("0" if len(parts) == 4 else "1")
    return chess.Board(" ".join(parts))


def fill(buf, i, board, move, cp):
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

    buf["stm"][i] = 1 if board.turn == chess.WHITE else 0
    buf["castling"][i] = castling
    buf["ep"][i] = board.ep_square if board.ep_square is not None else -1
    buf["from_sq"][i] = move.from_square
    buf["to_sq"][i] = move.to_square
    buf["promo"][i] = move.promotion if move.promotion else 0
    buf["cp"][i] = cp


def lines(path):
    """Stream decompressed lines without unpacking the whole archive.

    A partially downloaded file ends mid-frame, which raises ZstdError. That
    is expected here, not a failure - we just stop at the last whole line.
    """
    with open(path, "rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        buffer = b""
        while True:
            try:
                chunk = reader.read(1 << 20)
            except zstandard.ZstdError as exc:
                print(f"\nstream ends early ({exc}) - "
                      f"treating it as the end of a partial download")
                break
            if not chunk:
                break
            buffer += chunk
            *complete, buffer = buffer.split(b"\n")
            for line in complete:
                if line:
                    yield line


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    src = sys.argv[1]
    out_path = sys.argv[2]

    min_depth = MIN_DEPTH
    if "--min-depth" in sys.argv:
        min_depth = int(sys.argv[sys.argv.index("--min-depth") + 1])

    max_records = None
    if "--max" in sys.argv:
        max_records = int(sys.argv[sys.argv.index("--max") + 1])
    elif len(sys.argv) > 3 and not sys.argv[3].startswith("--"):
        max_records = int(sys.argv[3])

    print(f"minimum evaluation depth: {min_depth}")
    if max_records:
        print(f"stopping after {max_records:,} records")

    buf = np.zeros(BUFFER_SIZE, dtype=EVAL_RECORD)
    filled = written = 0
    seen = skipped_shallow = skipped_bad = 0
    start = time.time()

    # The database does not document whether cp is from White's point of view
    # or the side to move's. Rather than guess, correlate the sign against
    # material balance on a sample and find out.
    white_pov_hits = stm_pov_hits = sample = 0
    white_pov = True
    decided = False

    out = open(out_path, "wb")
    try:
        for raw in lines(src):
            if max_records is not None and written + filled >= max_records:
                break

            seen += 1
            if seen % 200_000 == 0:
                rate = seen / (time.time() - start)
                print(f"{seen:>11,} lines | {written + filled:>10,} records | "
                      f"{rate:6.0f} lines/s")

            try:
                entry = json.loads(raw)
            except ValueError:
                skipped_bad += 1
                continue

            ev, pv = best_pv(entry, min_depth)
            if pv is None:
                skipped_shallow += 1
                continue

            cp = score_of(pv)
            if cp is None:
                skipped_bad += 1
                continue

            try:
                board = board_from_fen(entry["fen"])
                move = chess.Move.from_uci(pv["line"].split()[0])
            except (ValueError, KeyError, IndexError):
                skipped_bad += 1
                continue

            if move not in board.legal_moves:
                skipped_bad += 1
                continue

            balance = material_balance(board)
            if not decided and abs(balance) >= 2:
                sample += 1
                if (cp > 0) == (balance > 0):
                    white_pov_hits += 1
                stm_sign = 1 if board.turn == chess.WHITE else -1
                if (cp > 0) == ((balance * stm_sign) > 0):
                    stm_pov_hits += 1
                if sample >= PERSPECTIVE_SAMPLE:
                    white_pov = white_pov_hits >= stm_pov_hits
                    decided = True
                    print(f"\nsign convention: cp appears to be from "
                          f"{'White' if white_pov else 'the side to move'}"
                          f"'s point of view")
                    print(f"  (white-pov agreement {white_pov_hits/sample*100:.1f}%, "
                          f"stm-pov agreement {stm_pov_hits/sample*100:.1f}%)\n")

            # Store everything from the side to move's view, matching how the
            # value head is trained.
            stored = cp
            if white_pov and board.turn == chess.BLACK:
                stored = -cp

            fill(buf, filled, board, move, stored)
            filled += 1
            if filled == BUFFER_SIZE:
                out.write(buf.tobytes())
                written += filled
                filled = 0

        if filled:
            out.write(buf[:filled].tobytes())
            written += filled
    finally:
        out.close()

    if not decided and sample:
        print("\nwarning: too few positions to settle the sign convention; "
              "assumed White's point of view")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  lines read       : {seen:,}")
    print(f"  too shallow      : {skipped_shallow:,}")
    print(f"  unparseable      : {skipped_bad:,}")
    print(f"  records written  : {written:,}")
    print(f"  file size        : {written * EVAL_RECORD.itemsize / 1e6:.0f} MB")

    data = np.memmap(out_path, dtype=EVAL_RECORD, mode="r")
    if len(data):
        cp = data["cp"].astype(np.float32)
        print(f"\n  score spread: mean {cp.mean():+.0f}cp, "
              f"median {np.median(cp):+.0f}cp")
        print(f"  {(np.abs(cp) < 50).mean()*100:.0f}% near-equal, "
              f"{(np.abs(cp) > 500).mean()*100:.0f}% decisive")


if __name__ == "__main__":
    main()
