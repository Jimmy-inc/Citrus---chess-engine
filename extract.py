"""
extract.py - turn a Lichess PGN file into compact binary training records.

Usage:
    python extract.py data/lichess_elite_2024-01.pgn data/positions.bin
    python extract.py data/lichess_elite_2024-01.pgn data/positions.bin 5000

The optional third argument caps how many games to read, which is handy
for a quick test run before you commit to the full file.
"""

import sys
import time
import random

import numpy as np
import chess
import chess.pgn

# ---------------------------------------------------------------- settings

MIN_ELO = 2000          # both players must be at least this
SKIP_PLIES = 8          # ignore the first N half-moves (opening book)
MIN_GAME_PLIES = 20     # ignore very short games
DROP_TIME_FORFEITS = True
BUFFER_SIZE = 100_000   # records held in memory before flushing to disk

# ---------------------------------------------------------------- format

# One record per position. 71 bytes each.
#   pieces   : 64 squares, 0 = empty, 1-6 = white P N B R Q K, 7-12 = black
#   stm      : 1 if white to move, 0 if black
#   castling : 4 bits - white kingside, white queenside, black KS, black QS
#   ep       : en passant target square, or -1
#   from_sq  : square the played move started on
#   to_sq    : square it ended on
#   promo    : promotion piece type (0 = none, 2=N 3=B 4=R 5=Q)
#   result   : +1 white won, 0 draw, -1 black won
RECORD = np.dtype([
    ("pieces",   np.int8, (64,)),
    ("stm",      np.int8),
    ("castling", np.int8),
    ("ep",       np.int8),
    ("from_sq",  np.int8),
    ("to_sq",    np.int8),
    ("promo",    np.int8),
    ("result",   np.int8),
])

PIECE_ID = {
    (chess.PAWN, True): 1,  (chess.KNIGHT, True): 2,  (chess.BISHOP, True): 3,
    (chess.ROOK, True): 4,  (chess.QUEEN, True): 5,   (chess.KING, True): 6,
    (chess.PAWN, False): 7, (chess.KNIGHT, False): 8, (chess.BISHOP, False): 9,
    (chess.ROOK, False): 10, (chess.QUEEN, False): 11, (chess.KING, False): 12,
}

RESULT_MAP = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}

# ---------------------------------------------------------------- helpers


def parse_elo(value):
    """Lichess writes '?' for unrated or missing ratings."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def game_is_usable(headers):
    if headers.get("Result") not in RESULT_MAP:
        return False
    if DROP_TIME_FORFEITS and headers.get("Termination") == "Time forfeit":
        return False
    white = parse_elo(headers.get("WhiteElo"))
    black = parse_elo(headers.get("BlackElo"))
    if white is None or black is None:
        return False
    return white >= MIN_ELO and black >= MIN_ELO


def write_position(buf, i, board, move, result):
    """Fill row i of the buffer from the current board and the move played."""
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
    buf["result"][i] = result


def verify(path):
    """Read a random record back and print it, to prove the file round-trips."""
    data = np.fromfile(path, dtype=RECORD)
    if len(data) == 0:
        print("No records written.")
        return

    rec = data[random.randrange(len(data))]
    board = chess.Board(None)  # empty board
    for square, piece_id in enumerate(rec["pieces"]):
        if piece_id:
            colour = piece_id <= 6
            piece_type = piece_id if colour else piece_id - 6
            board.set_piece_at(square, chess.Piece(int(piece_type), bool(colour)))
    board.turn = chess.WHITE if rec["stm"] else chess.BLACK

    move = chess.Move(
        int(rec["from_sq"]),
        int(rec["to_sq"]),
        promotion=int(rec["promo"]) or None,
    )

    print("\n--- random record, decoded back ---")
    print(board)
    print(f"{'White' if rec['stm'] else 'Black'} to move, played {move.uci()}, "
          f"game result {int(rec['result']):+d}")
    legal = move in board.legal_moves
    print(f"move is legal in reconstructed position: {legal}")
    if not legal:
        print("(castling rights are not restored here, so castling moves "
              "may show as illegal - that is expected)")


# ---------------------------------------------------------------- main


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
    games_read = 0
    games_kept = 0
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

            if games_read % 5000 == 0:
                rate = games_read / (time.time() - start)
                print(f"{games_read:>8,} games read | {games_kept:>8,} kept | "
                      f"{written + filled:>10,} positions | {rate:5.0f} games/s")

            if not game_is_usable(game.headers):
                continue

            moves = list(game.mainline_moves())
            if len(moves) < MIN_GAME_PLIES:
                continue

            result = RESULT_MAP[game.headers["Result"]]
            board = game.board()
            games_kept += 1

            for ply, move in enumerate(moves):
                if ply >= SKIP_PLIES:
                    write_position(buf, filled, board, move, result)
                    filled += 1
                    if filled == BUFFER_SIZE:
                        out.write(buf.tobytes())
                        written += filled
                        filled = 0
                board.push(move)

        if filled:
            out.write(buf[:filled].tobytes())
            written += filled
    finally:
        pgn.close()
        out.close()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  games read   : {games_read:,}")
    print(f"  games kept   : {games_kept:,}")
    print(f"  positions    : {written:,}")
    print(f"  file size    : {written * RECORD.itemsize / 1e6:.0f} MB")

    verify(out_path)


if __name__ == "__main__":
    main()
