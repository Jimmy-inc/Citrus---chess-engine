"""
convert_lc0.py - turn Leela training chunks into records this project can train on.

What makes these worth the trouble: every position carries the full MCTS
visit distribution over moves, not just the one move an engine preferred.
That is the real AlphaZero policy target, and it says far more than a single
label - it tells the network which alternatives were nearly as good.

Three things have to be reconciled.

  Move indexing.  Leela numbers moves 0..1857 in its own order; this project
  uses AlphaZero's 4672 (origin square, move type) scheme. The mapping is
  built at load time from Leela's own move list, so it cannot drift.

  Board orientation.  Leela packs bitboards little-endian, so unpacking the
  bytes yields squares with the files reversed. That gets undone here, and
  the self-test below catches it if it is ever wrong.

  History.  Each record holds eight positions; only the most recent is kept,
  since this network takes one position.

Leela's planes are already written from the mover's point of view, which is
the same convention this project canonicalized to, so no flipping is needed.

Usage:
    python convert_lc0.py data/lc0chunks data/lc0.bin
    python convert_lc0.py data/lc0chunks data/lc0.bin --max 20000000
    python convert_lc0.py --self-test data/lc0chunks

Options:
    --max       stop after this many records            (default no limit)
    --top       policy entries kept per position           (default 30)
    --value     which value target: result, best or root  (default result)
    --self-test decode a few records and check them against python-chess
"""

import glob
import gzip
import os
import struct
import sys
import time

from collections import Counter

import numpy as np
import chess

from encoding_v2 import move_index, POLICY_SIZE

V6_SIZE = 8356
# V5 is the same layout up to offset 8296, so the planes, policy, castling
# and best_q all sit where V6 puts them. The one difference that matters is
# the game result: V6 stores it as a float at 8308, V5 as a single signed
# byte at 8279.
V5_SIZE = 8308
V5_RESULT_AT = 8279
V6_PROBS_AT = 8
V6_PLANES_AT = 7440
V6_CASTLE_AT = 8272
V6_STM_AT = 8276
V6_RULE50_AT = 8277
V6_ROOT_Q_AT = 8280
V6_BEST_Q_AT = 8284
V6_RESULT_Q_AT = 8308

POLICY_ENTRIES = 1858
TOP_K = 30

# 64 signed bytes of piece codes, castling, the two value targets, and a
# sparse policy: the most visited moves and their shares.
def make_dtype(top_k):
    return np.dtype([
        ("pieces",   np.int8, (64,)),
        ("castling", np.int8),
        ("n_moves",  np.uint8),
        ("best_q",   np.float16),
        ("result_q", np.float16),
        ("root_q",   np.float16),
        ("pad",      np.int8, (1,)),
        ("idx",      np.uint16, (top_k,)),
        ("prob",     np.float16, (top_k,)),
    ])


# ---------------------------------------------------------------- mapping


def build_index_map():
    """
    Leela move number -> this project's policy index.

    Leela's list is UCI strings in its own fixed order. Parsing each one and
    asking encoding_v2 where it belongs gives the translation, so the two
    schemes stay consistent even if either changes.
    """
    try:
        import policy_index
    except ImportError:
        print("policy_index.py not found. Fetch Leela's move list first:\n"
              "  curl -O https://raw.githubusercontent.com/LeelaChessZero"
              "/lczero-training/master/tf/policy_index.py")
        sys.exit(1)

    moves = policy_index.policy_index
    if len(moves) != POLICY_ENTRIES:
        print(f"expected {POLICY_ENTRIES} moves, found {len(moves)}")
        sys.exit(1)

    mapping = np.full(POLICY_ENTRIES, -1, dtype=np.int32)
    unmapped = []
    for i, text in enumerate(moves):
        try:
            move = chess.Move.from_uci(text)
        except ValueError:
            unmapped.append(text)
            continue
        index = move_index(move.from_square, move.to_square,
                           move.promotion or 0)
        if index < 0:
            unmapped.append(text)
        else:
            mapping[i] = index

    # Leela writes castling as king-takes-rook (e1h1, e1a1); this project
    # writes it the usual way (e1g1, e1c1). The same strings are also legal
    # rook moves when a rook stands on e1, so the substitution can only be
    # made with the board in hand - see castle_swap below.
    # Each entry is leela move number -> (our policy index, rook square).
    castle_swap = {}
    for leela_text, ours_text, rook in (("e1h1", "e1g1", chess.H1),
                                        ("e1a1", "e1c1", chess.A1)):
        if leela_text in moves:
            ours = chess.Move.from_uci(ours_text)
            castle_swap[moves.index(leela_text)] = (
                move_index(ours.from_square, ours.to_square, 0), rook)

    return mapping, unmapped, castle_swap


# ---------------------------------------------------------------- planes


# Leela's bitboards unpack so that bit position j is square j directly -
# a1 first, h8 last, the same numbering python-chess uses. An earlier
# version of this file assumed a little-endian byte layout needing the files
# reversed, which silently produced mirrored boards; the self-test below is
# what caught it, so run it before trusting any conversion.
_SQUARE_OF_BIT = np.arange(64)

# Leela's plane order within a position: our pawn, knight, bishop, rook,
# queen, king, then theirs. This project codes white 1-6 and black 7-12,
# and after canonicalization "us" is white.
_PLANE_TO_PIECE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def decode_pieces(record):
    """The 64 piece codes for the most recent position in a record."""
    raw = np.frombuffer(record, dtype=np.uint8,
                        count=13 * 8, offset=V6_PLANES_AT)
    bits = np.unpackbits(raw).reshape(13, 64)

    pieces = np.zeros(64, dtype=np.int8)
    for plane in range(12):
        occupied = np.nonzero(bits[plane])[0]
        if len(occupied):
            pieces[_SQUARE_OF_BIT[occupied]] = _PLANE_TO_PIECE[plane]
    return pieces


def decode_castling(record):
    us_ooo, us_oo, them_ooo, them_oo = record[V6_CASTLE_AT:V6_CASTLE_AT + 4]
    rights = 0
    if us_oo:
        rights |= 1
    if us_ooo:
        rights |= 2
    if them_oo:
        rights |= 4
    if them_ooo:
        rights |= 8
    return rights


def board_from_record(record):
    """
    Rebuild a python-chess board from a record, for verification.

    En passant is not recoverable from the classical input format - it lives
    only in the history planes - so the reconstruction ignores it. That is
    also why the converter writes no en passant plane.
    """
    board = chess.Board(None)
    pieces = decode_pieces(record)
    for square in range(64):
        code = int(pieces[square])
        if code == 0:
            continue
        colour = chess.WHITE if code <= 6 else chess.BLACK
        kind = ((code - 1) % 6) + 1
        board.set_piece_at(square, chess.Piece(kind, colour))

    board.turn = chess.WHITE
    rights = decode_castling(record)
    fen_rights = ""
    if rights & 1:
        fen_rights += "K"
    if rights & 2:
        fen_rights += "Q"
    if rights & 4:
        fen_rights += "k"
    if rights & 8:
        fen_rights += "q"
    board.set_castling_fen(fen_rights or "-")
    return board


# ---------------------------------------------------------------- convert


def records_in(path, tally=None):
    """
    Yield (record, version) from one gzipped chunk.

    Both V6 and V5 are handled; older versions put the policy at a different
    offset entirely and are skipped. The tally, when given, records what was
    seen so a run that produces nothing can say why.
    """
    try:
        with gzip.open(path, "rb") as handle:
            blob = handle.read()
    except (OSError, EOFError):
        if tally is not None:
            tally["unreadable"] += 1
        return

    if len(blob) >= 8 and len(blob) % V6_SIZE == 0:
        size, version = V6_SIZE, 6
    elif len(blob) >= 8 and len(blob) % V5_SIZE == 0:
        size, version = V5_SIZE, 5
    else:
        if tally is not None:
            tally[f"unknown record size (len {len(blob)})"] += 1
        return

    for start in range(0, len(blob), size):
        record = blob[start:start + size]
        stored_version, input_format = struct.unpack("ii", record[:8])
        if stored_version != version:
            if tally is not None:
                tally[f"v{stored_version} in a v{version}-sized file"] += 1
            continue
        if input_format != 1:
            if tally is not None:
                tally[f"input_format {input_format}"] += 1
            continue
        if tally is not None:
            tally[f"v{version}"] += 1
        yield record, version


def convert(record, version, mapping, top_k, castle_swap, out):
    probs = np.frombuffer(record, dtype=np.float32,
                          count=POLICY_ENTRIES, offset=V6_PROBS_AT)

    live = np.nonzero(probs > 0)[0]
    if len(live) == 0:
        return False

    order = live[np.argsort(-probs[live])][:top_k]
    indices = mapping[order].copy()

    pieces_now = decode_pieces(record)

    # Substitute only where the position really is a castling one: our king
    # on e1 (code 6) and our rook (code 4) on the square Leela names.
    for slot, leela_index in enumerate(order):
        entry = castle_swap.get(int(leela_index))
        if entry is None:
            continue
        ours, rook_square = entry
        if pieces_now[4] == 6 and pieces_now[rook_square] == 4:
            indices[slot] = ours

    keep = indices >= 0
    order, indices = order[keep], indices[keep]
    if len(order) == 0:
        return False

    kept = probs[order]
    total = kept.sum()
    if total <= 0:
        return False
    kept = kept / total          # renormalise over what was kept

    out["pieces"] = pieces_now
    out["castling"] = decode_castling(record)
    out["n_moves"] = len(order)
    out["best_q"] = struct.unpack("f", record[V6_BEST_Q_AT:V6_BEST_Q_AT + 4])[0]
    out["root_q"] = struct.unpack("f", record[V6_ROOT_Q_AT:V6_ROOT_Q_AT + 4])[0]
    if version == 6:
        out["result_q"] = struct.unpack(
            "f", record[V6_RESULT_Q_AT:V6_RESULT_Q_AT + 4])[0]
    else:
        # V5's result is +1, 0 or -1 from the mover's point of view, which
        # is the same convention as V6's result_q, just coarser.
        out["result_q"] = float(
            struct.unpack("b", record[V5_RESULT_AT:V5_RESULT_AT + 1])[0])
    out["idx"][:] = 0
    out["prob"][:] = 0.0
    out["idx"][:len(indices)] = indices
    out["prob"][:len(kept)] = kept
    return True


def self_test(source):
    """
    Decode real records and check them against python-chess.

    The strong check is that every move Leela gave a visit share to is legal
    in the reconstructed position. If the board were mirrored, or the move
    mapping wrong, almost none of them would be.
    """
    mapping, unmapped, castle_swap = build_index_map()
    print(f"move mapping: {int((mapping >= 0).sum())}/{POLICY_ENTRIES} "
          f"Leela moves mapped")
    if unmapped:
        print(f"  unmapped examples: {unmapped[:8]}")

    import policy_index
    names = policy_index.policy_index

    files = sorted(glob.glob(os.path.join(source, "**", "*.gz"),
                             recursive=True))
    if not files:
        print(f"no .gz chunks under {source}")
        sys.exit(1)
    print(f"{len(files):,} chunk files found\n")

    checked = legal_ok = kings_ok = 0
    illegal_examples = []

    for path in files[:40]:
        for record, _version in records_in(path):
            board = board_from_record(record)
            probs = np.frombuffer(record, dtype=np.float32,
                                  count=POLICY_ENTRIES, offset=V6_PROBS_AT)
            live = np.nonzero(probs > 0)[0]
            if len(live) == 0:
                continue

            checked += 1
            if (len(board.pieces(chess.KING, chess.WHITE)) == 1
                    and len(board.pieces(chess.KING, chess.BLACK)) == 1):
                kings_ok += 1

            legal = {m.uci() for m in board.legal_moves}
            # Leela spells a queen promotion without the piece letter, while
            # python-chess always writes it. A move counts as legal if
            # either spelling is - not both, which is what an earlier
            # version of this check wrongly demanded.
            bad = []
            for i in live:
                text = names[i]
                if text in legal:
                    continue
                if len(text) == 4 and text + "q" in legal:
                    continue
                # king-takes-rook castling
                if text == "e1h1" and "e1g1" in legal:
                    continue
                if text == "e1a1" and "e1c1" in legal:
                    continue
                bad.append(text)

            if not bad:
                legal_ok += 1
            elif len(illegal_examples) < 3:
                illegal_examples.append((board.fen(), sorted(bad)[:6]))

            if checked >= 400:
                break
        if checked >= 400:
            break

    print(f"positions checked           : {checked}")
    print(f"  exactly two kings         : {kings_ok} "
          f"({kings_ok/checked*100:.1f}%)")
    print(f"  all visited moves legal   : {legal_ok} "
          f"({legal_ok/checked*100:.1f}%)")
    if illegal_examples:
        print("\n  positions where they were not:")
        for fen, bad in illegal_examples:
            print(f"    {fen}\n      {bad}")
        print("\n  A low legality rate means the board is being decoded "
              "wrongly - most likely the file mirroring.")
    else:
        print("\n  decoding looks correct")


def main():
    if "--self-test" in sys.argv:
        where = sys.argv[sys.argv.index("--self-test") + 1]
        self_test(where)
        return

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    source, destination = args[0], args[1]

    def opt(name, default, cast):
        if name in sys.argv:
            return cast(sys.argv[sys.argv.index(name) + 1])
        return default

    limit = opt("--max", 0, int)
    top_k = opt("--top", TOP_K, int)
    value_field = opt("--value", "result", str)

    mapping, unmapped, castle_swap = build_index_map()
    print(f"move mapping: {int((mapping >= 0).sum())}/{POLICY_ENTRIES} mapped")
    print(f"castling remaps: {len(castle_swap)}")
    if unmapped:
        print(f"  {len(unmapped)} unmapped, e.g. {unmapped[:5]}")

    dtype = make_dtype(top_k)
    print(f"record size: {dtype.itemsize} bytes, keeping top {top_k} moves")
    print(f"value target: {value_field}_q\n")

    files = sorted(glob.glob(os.path.join(source, "**", "*.gz"),
                             recursive=True))
    print(f"{len(files):,} chunk files\n")

    scratch = np.zeros(1, dtype=dtype)
    written = skipped = 0
    tally = Counter()
    start = time.time()

    with open(destination, "wb") as out:
        for number, path in enumerate(files, 1):
            for record, version in records_in(path, tally):
                if convert(record, version, mapping, top_k, castle_swap,
                           scratch[0]):
                    out.write(scratch.tobytes())
                    written += 1
                else:
                    skipped += 1

                if limit and written >= limit:
                    break

            if number % 500 == 0 or (limit and written >= limit):
                rate = written / max(time.time() - start, 1e-6)
                print(f"  {number:>7,}/{len(files):,} files | "
                      f"{written:>12,} records | {rate:7.0f}/s")
            if limit and written >= limit:
                break

    size = os.path.getsize(destination)
    print(f"\nDone in {(time.time()-start)/60:.1f} min")
    print(f"  records written : {written:,}")
    print(f"  skipped         : {skipped:,}")
    print(f"  file size       : {size/1e9:.2f} GB")

    if written == 0:
        print("\n  nothing was converted. What the files contained:")
        for label, count in tally.most_common(8):
            print(f"    {count:>10,}  {label}")


if __name__ == "__main__":
    main()
