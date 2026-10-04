"""
extract_engine.py - turn a TCEC archive PGN into the same 71-byte records
that extract.py produces, so the two datasets can be concatenated.

Usage:
    python extract_engine.py data/largest.pgn data/tcec.bin
    python extract_engine.py data/largest.pgn data/tcec.bin 2000

Differences from extract.py, all forced by how engine games are played:

  * Openings come from a forced book, so instead of skipping a fixed number
    of plies we skip to the "Book exit" marker in each game.
  * Chess960 games are excluded - castling there is encoded as king-takes-rook,
    which would pollute the from/to move vocabulary the policy head uses.
  * Games without Elo headers are dropped, since without them the strength
    filter cannot be applied.
  * Games that ended for technical reasons (abandoned, dropped connection,
    flagged) are dropped - the result does not reflect the play.
"""

import sys
import time

import numpy as np
import chess
import chess.pgn

from extract import RECORD, RESULT_MAP, parse_elo, write_position, verify

# ---------------------------------------------------------------- settings

MIN_ELO = 3200          # both engines must be at least this
FALLBACK_SKIP = 16      # plies to skip if a game has no book-exit marker
MIN_RECORDED_PLIES = 10 # need at least this many post-book moves
BUFFER_SIZE = 100_000

# Results here are technical, not chessic.
BAD_TERMINATIONS = {
    "abandoned",
    "stalled connection",
    "time forfeit",
    "time",
    "unterminated",
}


def game_is_usable(headers):
    if headers.get("Result") not in RESULT_MAP:
        return False

    variant = headers.get("Variant", "").strip().lower()
    if variant and variant not in ("normal", "standard", "chess"):
        return False

    termination = headers.get("Termination", "").strip().lower()
    if termination in BAD_TERMINATIONS:
        return False

    white = parse_elo(headers.get("WhiteElo"))
    black = parse_elo(headers.get("BlackElo"))
    if white is None or black is None:
        return False
    if white < MIN_ELO or black < MIN_ELO:
        return False

    return True


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    pgn_path = sys.argv[1]
    out_path = sys.argv[2]
    max_games = int(sys.argv[3]) if len(sys.argv) > 3 else None

    buf = np.zeros(BUFFER_SIZE, dtype=RECORD)
    filled = 0
    written = 0
    games_read = games_kept = 0
    no_marker = 0
    start = time.time()

    pgn = open(pgn_path, encoding="utf-8", errors="replace")
    out = open(out_path, "wb")

    try:
        while True:
            if max_games is not None and games_read >= max_games:
                break

            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games_read += 1

            if games_read % 2000 == 0:
                rate = games_read / (time.time() - start)
                print(f"{games_read:>7,} games read | {games_kept:>7,} kept | "
                      f"{written + filled:>9,} positions | {rate:5.0f} games/s")

            if not game_is_usable(game.headers):
                continue

            result = RESULT_MAP[game.headers["Result"]]

            nodes = list(game.mainline())
            if not nodes:
                continue

            has_marker = any("Book exit" in (n.comment or "") for n in nodes)
            if not has_marker:
                no_marker += 1

            board = game.board()
            in_book = has_marker          # if no marker, fall back to a ply count
            recorded = 0
            start_filled = filled

            for ply, node in enumerate(nodes):
                move = node.move
                if move is None:
                    break

                past_book = (not in_book) if has_marker else (ply >= FALLBACK_SKIP)
                if past_book:
                    if filled == BUFFER_SIZE:
                        out.write(buf.tobytes())
                        written += filled
                        filled = 0
                        start_filled = 0
                    write_position(buf, filled, board, move, result)
                    filled += 1
                    recorded += 1

                board.push(move)

                if in_book and "Book exit" in (node.comment or ""):
                    in_book = False

            # Too short to be worth keeping - roll the buffer back.
            if recorded < MIN_RECORDED_PLIES and start_filled <= filled:
                filled = start_filled
            else:
                games_kept += 1

        if filled:
            out.write(buf[:filled].tobytes())
            written += filled
    finally:
        pgn.close()
        out.close()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  games read        : {games_read:,}")
    print(f"  games kept        : {games_kept:,}")
    print(f"  without book mark : {no_marker:,}")
    print(f"  positions         : {written:,}")
    print(f"  file size         : {written * RECORD.itemsize / 1e6:.0f} MB")

    verify(out_path)


if __name__ == "__main__":
    main()
