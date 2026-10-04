"""
encoding_v2.py - the board and move representation for the version two net.

Two changes from the original encoding, both of which make the network's job
smaller rather than making the network bigger.

Canonicalization
----------------
The old encoding always described the board from White's point of view and
added a plane saying whose turn it was. That forces the net to learn every
pattern twice - once as White, once mirrored as Black - from data that is
effectively split in half. Here the board is flipped and the colours swapped
whenever Black is to move, so the net only ever sees "me at the bottom,
opponent at the top". Same data, half the problem, and the side-to-move
plane is no longer needed at all.

Convolutional policy
--------------------
The old policy head was a dense layer over all 4096 from/to pairs, which
held 8.4M of the old net's 10.3M parameters and learned each pair in
isolation - a knight move on d4 taught it nothing about the same move on e5.
This uses AlphaZero's scheme instead: 73 planes of 8x8, where a plane means
a *kind* of move (this direction, this far) and the square says where it
starts from. Convolution then shares what it learns across the board.

    planes  0-55   queen-like moves: 8 directions x 7 distances
    planes 56-63   the 8 knight moves
    planes 64-72   underpromotions: 3 directions x knight/bishop/rook

Queen promotions are just the pawn move that reaches the last rank, so they
need no plane of their own.

Self-test:
    python encoding_v2.py
"""

import numpy as np
import chess
import torch

# 12 piece planes, 4 castling rights, 1 en passant. No side-to-move plane -
# after canonicalization it would be constant.
PLANES = 17

POLICY_PLANES = 73
POLICY_SIZE = 64 * POLICY_PLANES          # 4672

# Order matters only in that it stays fixed: N, NE, E, SE, S, SW, W, NW.
QUEEN_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1),
                    (0, -1), (-1, -1), (-1, 0), (-1, 1)]

KNIGHT_OFFSETS = [(1, 2), (2, 1), (2, -1), (1, -2),
                  (-1, -2), (-2, -1), (-2, 1), (-1, 2)]

UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _build_move_lookup():
    """
    A table from (from square, to square, promotion) to policy index.

    Building it once at import time means training and search both do a
    single array lookup rather than recomputing geometry per move.
    """
    table = np.full((64, 64, 6), -1, dtype=np.int16)

    for from_square in range(64):
        from_file = chess.square_file(from_square)
        from_rank = chess.square_rank(from_square)

        for to_square in range(64):
            if to_square == from_square:
                continue
            to_file = chess.square_file(to_square)
            to_rank = chess.square_rank(to_square)
            file_delta = to_file - from_file
            rank_delta = to_rank - from_rank

            plane = None

            if (abs(file_delta), abs(rank_delta)) in ((1, 2), (2, 1)):
                offset = (file_delta, rank_delta)
                plane = 56 + KNIGHT_OFFSETS.index(offset)
            elif file_delta == 0 or rank_delta == 0 or \
                    abs(file_delta) == abs(rank_delta):
                distance = max(abs(file_delta), abs(rank_delta))
                step = (np.sign(file_delta), np.sign(rank_delta))
                direction = QUEEN_DIRECTIONS.index((int(step[0]), int(step[1])))
                plane = direction * 7 + (distance - 1)

            if plane is None:
                continue

            index = from_square * POLICY_PLANES + plane
            # No promotion, or a queen promotion, uses the plain move plane.
            table[from_square, to_square, 0] = index
            table[from_square, to_square, chess.QUEEN] = index

            # Underpromotions only exist for a pawn stepping onto the last
            # rank, one square forward or one square diagonally.
            if to_rank == 7 and rank_delta == 1 and abs(file_delta) <= 1:
                for promotion_slot, piece in enumerate(UNDERPROMOTIONS):
                    under_plane = 64 + promotion_slot * 3 + (file_delta + 1)
                    table[from_square, to_square, piece] = (
                        from_square * POLICY_PLANES + under_plane)

    return table


MOVE_LOOKUP = _build_move_lookup()


# The same table as plain Python ints, for the search's hot path: indexing a
# list is several times cheaper than indexing a numpy array and converting.
_LOOKUP_FLAT = MOVE_LOOKUP.reshape(-1).tolist()
_PROMOS = MOVE_LOOKUP.shape[2]


# Lichess's evaluation database writes castling king-onto-rook (e1h1), and
# every net trained on it learned castling at that output. python-chess
# gives the search e1g1, a different output the net never learned castling
# on. True looks castling up where the net learned it; False is the old,
# broken lookup, kept only to measure the difference.
CASTLE_ONTO_ROOK = True


def policy_indices(board, moves):
    """
    (moves, policy indices) for the given legal moves, in the given order.
    """
    flip = 56 if board.turn == chess.BLACK else 0
    lookup = _LOOKUP_FLAT
    # A king on e1 moving two squares is castling and nothing else.
    castle_from = -1
    if CASTLE_ONTO_ROOK and board.castling_rights:
        king = board.king(board.turn)
        if king is not None and king ^ flip == 4:
            castle_from = 4
    kept, indices = [], []
    for move in moves:
        from_square = move.from_square ^ flip
        to_square = move.to_square ^ flip
        if from_square == castle_from and (to_square == 6 or to_square == 2):
            to_square = 7 if to_square == 6 else 0
        index = lookup[((from_square << 6 | to_square) * _PROMOS)
                       + (move.promotion or 0)]
        if index >= 0:
            kept.append(move)
            indices.append(index)
    return kept, indices


def move_index(from_square, to_square, promotion=0):
    """Policy index for a move, or -1 if the geometry is not representable."""
    return int(MOVE_LOOKUP[from_square, to_square, promotion or 0])


# ---------------------------------------------------------------- flipping


def flip_square(square):
    """Mirror a square vertically: a1 becomes a8, e4 becomes e5."""
    return square ^ 56


# Swapping colours means adding or subtracting 6 in the piece encoding,
# where 1-6 are White pawn..king and 7-12 are Black's.
_SWAP_PIECE = np.zeros(13, dtype=np.int8)
for _piece in range(1, 7):
    _SWAP_PIECE[_piece] = _piece + 6
    _SWAP_PIECE[_piece + 6] = _piece

# Castling bits are white-kingside, white-queenside, black-kingside,
# black-queenside. Swapping colours swaps the halves.
_SWAP_CASTLING = np.zeros(16, dtype=np.int8)
for _rights in range(16):
    _SWAP_CASTLING[_rights] = (((_rights >> 2) & 0b11)
                               | ((_rights & 0b11) << 2))

# Vertical mirror as an index permutation, so a whole batch flips at once.
_FLIP_INDEX = np.arange(64) ^ 56


def canonicalize(pieces, stm, castling, ep, from_sq=None, to_sq=None):
    """
    Rewrite a batch of positions from the mover's point of view.

    Positions where White is already to move pass through unchanged. Where
    Black is to move, the board is mirrored and the colours swapped, so
    afterwards "my pieces" are always the ones encoded 1-6 and always move
    up the board.

    Returns the rewritten arrays; move squares are returned only if given.
    """
    black_to_move = (stm == 0)

    pieces = pieces.copy()
    castling = castling.copy()
    ep = ep.copy()

    if black_to_move.any():
        rows = np.nonzero(black_to_move)[0]
        pieces[rows] = _SWAP_PIECE[pieces[rows][:, _FLIP_INDEX]]
        castling[rows] = _SWAP_CASTLING[castling[rows]]
        has_ep = ep[rows] >= 0
        ep[rows] = np.where(has_ep, ep[rows] ^ 56, -1)

    if from_sq is None:
        return pieces, castling, ep

    from_sq = from_sq.copy()
    to_sq = to_sq.copy()
    if black_to_move.any():
        rows = np.nonzero(black_to_move)[0]
        from_sq[rows] ^= 56
        to_sq[rows] ^= 56

    return pieces, castling, ep, from_sq, to_sq


# ---------------------------------------------------------------- planes


_EYE13 = {np.float32: np.eye(13, dtype=np.float32),
          np.uint8: np.eye(13, dtype=np.uint8)}


def planes_from_pieces(pieces, castling, ep, dtype=np.float32):
    """
    Expand canonical integer fields into the 17 input planes.

    The planes are all zeros and ones, so the search asks for uint8: a
    quarter of the bytes to send to the GPU, and exactly the same values
    once converted to float there.
    """
    count = len(pieces)

    occupied = _EYE13[dtype][pieces.astype(np.int64)][:, :, 1:]    # (B,64,12)
    occupied = occupied.transpose(0, 2, 1).reshape(count, 12, 8, 8)

    extra = np.zeros((count, 5, 8, 8), dtype=dtype)
    rights = castling.astype(np.int32)
    for bit in range(4):
        extra[:, bit] = ((rights >> bit) & 1).astype(dtype)[:, None, None]

    ep_plane = np.zeros((count, 64), dtype=dtype)
    rows = np.nonzero(ep >= 0)[0]
    ep_plane[rows, ep[rows]] = 1.0
    extra[:, 4] = ep_plane.reshape(count, 8, 8)

    return np.concatenate([occupied, extra], axis=1)


_PIECE_ID = {
    (chess.PAWN, True): 1, (chess.KNIGHT, True): 2, (chess.BISHOP, True): 3,
    (chess.ROOK, True): 4, (chess.QUEEN, True): 5, (chess.KING, True): 6,
    (chess.PAWN, False): 7, (chess.KNIGHT, False): 8, (chess.BISHOP, False): 9,
    (chess.ROOK, False): 10, (chess.QUEEN, False): 11, (chess.KING, False): 12,
}


def encode_board(board):
    """A single live position as a canonical (1, 17, 8, 8) tensor."""
    pieces = np.zeros((1, 64), dtype=np.int8)
    for square, piece in board.piece_map().items():
        pieces[0, square] = _PIECE_ID[(piece.piece_type, piece.color)]

    rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        rights |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        rights |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        rights |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        rights |= 8

    stm = np.array([1 if board.turn == chess.WHITE else 0], dtype=np.int8)
    castling = np.array([rights], dtype=np.int8)
    ep = np.array([board.ep_square if board.ep_square is not None else -1],
                  dtype=np.int8)

    pieces, castling, ep = canonicalize(pieces, stm, castling, ep)
    return torch.from_numpy(planes_from_pieces(pieces, castling, ep))


_BIT_PIECE_ID = np.arange(1, 13, dtype=np.int8)[None, :, None]


def encode_boards(boards, dtype=np.float32):
    """
    A batch of live positions as one (B, 17, 8, 8) tensor - identical to
    concatenating encode_board over them, but built from the bitboards in a
    handful of array operations rather than a Python loop over every square.
    """
    count = len(boards)
    masks = []
    stm = np.empty(count, dtype=np.int8)
    castling = np.zeros(count, dtype=np.int8)
    ep = np.empty(count, dtype=np.int8)

    for i, board in enumerate(boards):
        white, black = board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK]
        for colour in (white, black):
            masks += (board.pawns & colour, board.knights & colour,
                      board.bishops & colour, board.rooks & colour,
                      board.queens & colour, board.kings & colour)
        stm[i] = 1 if board.turn == chess.WHITE else 0
        # No rights on the board means all four answers are no - the common
        # case, and it saves four method calls.
        if board.castling_rights:
            castling[i] = (board.has_kingside_castling_rights(chess.WHITE)
                           | board.has_queenside_castling_rights(chess.WHITE) << 1
                           | board.has_kingside_castling_rights(chess.BLACK) << 2
                           | board.has_queenside_castling_rights(chess.BLACK) << 3)
        ep[i] = board.ep_square if board.ep_square is not None else -1

    # Each bitboard's bit n is square n, so unpacking the little-endian bytes
    # lays every piece type out as a 64-square row in one step.
    bits = np.unpackbits(np.array(masks, dtype=np.uint64).view(np.uint8)
                         .reshape(count, 12, 8), axis=2, bitorder="little")
    pieces = (bits * _BIT_PIECE_ID).sum(axis=1).astype(np.int8)

    pieces, castling, ep = canonicalize(pieces, stm, castling, ep)
    return torch.from_numpy(planes_from_pieces(pieces, castling, ep, dtype))


def legal_policy_indices(board):
    """
    {move: policy index} for the legal moves, in canonical orientation.

    The search needs this to mask the policy output down to what is actually
    playable, and to map the chosen index back to a real move.
    """
    return dict(zip(*policy_indices(board, list(board.legal_moves))))


# ---------------------------------------------------------------- self-test


def main():
    passed = total = 0

    def check(name, condition, detail=""):
        nonlocal passed, total
        total += 1
        passed += bool(condition)
        print(f"  {'ok  ' if condition else 'FAIL'}  {name}"
              f"{('  ' + detail) if detail else ''}")

    print("move indexing")
    check("policy size", POLICY_SIZE == 4672, f"{POLICY_SIZE}")

    # Every legal move in a varied set of positions must map to a distinct,
    # in-range index. A collision would mean two moves sharing an output.
    positions = [
        chess.Board(),
        chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"),
        chess.Board("8/PPPPPPPP/8/8/8/8/pppppppp/K6k w - - 0 1"),
        chess.Board("8/2P5/8/8/8/8/8/K6k w - - 0 1"),
        chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"),
    ]
    collisions = out_of_range = unmapped = 0
    checked = 0
    for board in positions:
        mapping = legal_policy_indices(board)
        unmapped += len(list(board.legal_moves)) - len(mapping)
        seen = {}
        for move, index in mapping.items():
            checked += 1
            if not 0 <= index < POLICY_SIZE:
                out_of_range += 1
            if index in seen:
                collisions += 1
            seen[index] = move

    check("every legal move maps", unmapped == 0, f"{unmapped} unmapped")
    check("indices in range", out_of_range == 0, f"{checked} moves checked")
    check("no two moves share an index", collisions == 0,
          f"{collisions} collisions")

    # Castling, looked up where the training data put it, must still be
    # distinct from every other move in a position where both sides can.
    global CASTLE_ONTO_ROOK
    saved, CASTLE_ONTO_ROOK = CASTLE_ONTO_ROOK, True
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    mapping = legal_policy_indices(board)
    castles = {board.san(m): i for m, i in mapping.items()
               if board.is_castling(m)}
    CASTLE_ONTO_ROOK = saved
    check("castling at the king-onto-rook outputs, no collisions",
          castles == {"O-O": move_index(4, 7), "O-O-O": move_index(4, 0)}
          and len(set(mapping.values())) == len(mapping), f"{castles}")

    # Underpromotions must be distinguishable from queen promotions.
    board = chess.Board("8/2P5/8/8/8/8/8/K6k w - - 0 1")
    mapping = legal_policy_indices(board)
    promos = {m.promotion: i for m, i in mapping.items() if m.promotion}
    check("four promotion pieces, four indices",
          len(set(promos.values())) == 4, f"{sorted(promos.values())}")

    print("\ncanonicalization")

    # The same position with colours reversed and the board flipped must
    # produce identical planes - that is the whole point.
    white_view = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    black_view = chess.Board("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1")
    same = torch.equal(encode_board(white_view), encode_board(black_view))
    check("mirrored positions encode identically", same)

    # A position with Black to move must not encode the same as its
    # unflipped self - otherwise nothing is happening.
    plain = chess.Board("4k3/4p3/8/8/8/8/8/4K3 w - - 0 1")
    differs = not torch.equal(encode_board(plain), encode_board(black_view))
    check("flipping actually changes something", differs)

    check("plane count", encode_board(white_view).shape == (1, PLANES, 8, 8),
          str(tuple(encode_board(white_view).shape)))

    # Castling rights have to travel with the colours.
    both = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    pieces = np.zeros((1, 64), dtype=np.int8)
    for square, piece in both.piece_map().items():
        pieces[0, square] = _PIECE_ID[(piece.piece_type, piece.color)]
    _, castling, _ = canonicalize(pieces, np.array([0], dtype=np.int8),
                                  np.array([0b1111], dtype=np.int8),
                                  np.array([-1], dtype=np.int8))
    check("castling rights swap with colours", castling[0] == 0b1111,
          f"{castling[0]:04b}")

    one_side = canonicalize(pieces, np.array([0], dtype=np.int8),
                            np.array([0b0011], dtype=np.int8),
                            np.array([-1], dtype=np.int8))[1]
    check("white-only rights become black-only", one_side[0] == 0b1100,
          f"{one_side[0]:04b}")

    print(f"\n{passed}/{total} checks passed")


if __name__ == "__main__":
    main()
