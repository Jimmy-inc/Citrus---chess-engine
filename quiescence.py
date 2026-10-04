"""
quiescence.py - resolve tactics before the network is asked to judge a
position, using a real quiescence search rather than a greedy sequence.

Why the first attempt failed
----------------------------
The earlier version played out whichever captures looked good by static
exchange evaluation and handed the network whatever position it landed on.
Three things were wrong with that:

  * it followed one line instead of searching alternatives;
  * the side to move was treated as obliged to capture, when declining is
    always an option and is often better;
  * a checkmate at the end of that unforced line was recorded as a forced
    win, which is how the search talked itself into wins that were not real.

The correct recursion is a small negamax with the option of standing pat:

    value(P) = max( eval(P), max over captures c of -value(P after c) )

That first term is the refusal to capture. Without it, a sequence that is
good for the opponent gets scored as though the mover had no choice.

Cost
----
A quiescence search wants several evaluations per leaf, and here each one is
a network call. The saving grace is that the caller hands us many leaves at
once, so every level is evaluated in a single batch - and most positions
have no winning captures at all, so they cost nothing beyond the evaluation
they needed anyway.

Self-test:
    python quiescence.py
"""

import chess

# Centipawn values used only for exchange arithmetic, never for evaluation.
SEE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

DEFAULT_DEPTH = 2
MAX_CAPTURES = 4       # widest branching allowed at a quiescence node
MAX_EVASIONS = 6       # ditto when the side to move is in check


# ---------------------------------------------------------------- exchanges


def _least_valuable_attacker(board, square, colour):
    """Square of the cheapest piece of `colour` attacking `square`."""
    best_square = None
    best_value = None
    for from_square in board.attackers(colour, square):
        piece_type = board.piece_type_at(from_square)
        if piece_type is None:
            continue
        value = SEE_VALUES[piece_type]
        if best_value is None or value < best_value:
            best_value = value
            best_square = from_square
    return best_square


def _recapture_gain(board, target):
    """
    Material the side to move can win by continuing the exchange on `target`.

    Zero is always available - nobody is forced to recapture - which is what
    the max(0, ...) represents, and what makes this an evaluation of best
    play rather than of a forced sequence.
    """
    from_square = _least_valuable_attacker(board, target, board.turn)
    if from_square is None:
        return 0

    on_target = board.piece_type_at(target)
    if on_target is None:
        return 0
    target_value = SEE_VALUES[on_target]

    move = chess.Move(from_square, target)
    if (board.piece_type_at(from_square) == chess.PAWN
            and chess.square_rank(target) in (0, 7)):
        move = chess.Move(from_square, target, promotion=chess.QUEEN)

    if move not in board.legal_moves:
        return 0            # pinned attacker, or the promotion is blocked

    board.push(move)
    gain = max(0, target_value - _recapture_gain(board, target))
    board.pop()
    return gain


def see(board, move):
    """
    Static exchange evaluation: centipawns won by playing this capture,
    assuming both sides then recapture optimally.
    """
    if not board.is_capture(move):
        return 0

    if board.is_en_passant(move):
        captured_value = SEE_VALUES[chess.PAWN]
    else:
        captured = board.piece_type_at(move.to_square)
        if captured is None:
            return 0
        captured_value = SEE_VALUES[captured]

    working = board.copy(stack=False)
    working.push(move)
    return captured_value - _recapture_gain(working, move.to_square)


# ---------------------------------------------------------------- search


def terminal_value(board, contempt):
    """A finished position's value, from the side to move's perspective."""
    if board.is_checkmate():
        return -1.0
    # Repetition and the fifty-move rule cannot arise inside a capture
    # sequence, so the cheap checks are enough here.
    if board.is_stalemate() or board.is_insufficient_material():
        return -contempt
    return None


def tactical_moves(board):
    """
    What is worth searching at a quiescence node.

    In check, every legal move - the side to move has no choice about
    dealing with it, and pretending otherwise is how you miss being mated.
    Otherwise, only captures that do not lose material, best first.
    """
    if board.is_check():
        return list(board.legal_moves)[:MAX_EVASIONS]

    scored = []
    for move in board.legal_moves:
        if not board.is_capture(move):
            continue
        gain = see(board, move)
        if gain >= 0:
            scored.append((gain, move))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [move for _, move in scored[:MAX_CAPTURES]]


def qsearch(evaluate, boards, depth=DEFAULT_DEPTH, contempt=0.0,
            stand_pat=None):
    """
    Quiescence values for a whole batch of positions at once.

    evaluate:  callable taking a list of boards and returning a list of
               values, each from its own board's side-to-move perspective.
               This is where the network call happens - once per level.
    boards:    positions to evaluate.
    depth:     plies of captures to search. 0 is a plain evaluation.
    stand_pat: evaluations for `boards` the caller already has, so the level
               it just computed is not computed twice.

    Returns one value per board, in the same order.
    """
    values = [None] * len(boards)

    # Finished positions need no evaluation at all.
    pending = []
    pending_index = []
    for i, board in enumerate(boards):
        settled = terminal_value(board, contempt)
        if settled is not None:
            values[i] = settled
        elif stand_pat is not None:
            values[i] = stand_pat[i]
        else:
            pending.append(board)
            pending_index.append(i)

    if pending:
        for slot, value in zip(pending_index, evaluate(pending)):
            values[slot] = value

    if depth <= 0:
        return values

    # Gather every tactical continuation across every board, so the next
    # level is a single batch rather than one call per position.
    children = []
    parent_of = []
    for i, board in enumerate(boards):
        if terminal_value(board, contempt) is not None:
            continue
        for move in tactical_moves(board):
            child = board.copy(stack=False)
            child.push(move)
            children.append(child)
            parent_of.append(i)

    if not children:
        return values

    child_values = qsearch(evaluate, children, depth - 1, contempt)

    best = {}
    for j, i in enumerate(parent_of):
        score = -child_values[j]        # child value is the opponent's view
        if i not in best or score > best[i]:
            best[i] = score

    for i, score in best.items():
        if boards[i].is_check():
            # Standing pat is not on offer when you are in check, so the
            # search result replaces the static evaluation rather than
            # competing with it.
            values[i] = score
        else:
            values[i] = max(values[i], score)

    return values


# ---------------------------------------------------------------- self-test


def _material_eval(board):
    """Stand-in evaluator for testing: material only, roughly in tanh units."""
    total = 0
    for piece in board.piece_map().values():
        if piece.piece_type == chess.KING:
            continue
        value = SEE_VALUES[piece.piece_type]
        total += value if piece.color == board.turn else -value
    return max(-1.0, min(1.0, total / 1000.0))


def main():
    passed = total = 0

    def check(name, got, expected, tolerance=1e-6):
        nonlocal passed, total
        total += 1
        ok = abs(got - expected) <= tolerance
        passed += ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}: got {got:+.3f}, "
              f"expected {expected:+.3f}")

    calls = [0]

    def evaluate(boards):
        calls[0] += len(boards)
        return [_material_eval(b) for b in boards]

    print("static exchange evaluation")
    for name, fen, uci, expected in [
        ("free pawn", "4k3/8/8/3p4/8/8/8/3RK3 w - - 0 1", "d1d5", 100),
        ("defended pawn", "4k3/8/2p5/3p4/8/8/8/3RK3 w - - 0 1", "d1d5", -400),
        ("defended queen", "4k3/8/2p5/3q4/8/8/8/3RK3 w - - 0 1", "d1d5", 400),
    ]:
        total += 1
        got = see(chess.Board(fen), chess.Move.from_uci(uci))
        ok = got == expected
        passed += ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {got:+d} "
              f"(expected {expected:+d})")

    print("\nquiescence search")

    quiet = chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
    check("quiet position equals its static eval",
          qsearch(evaluate, [quiet], depth=2)[0], _material_eval(quiet))

    free = chess.Board("4k3/8/8/3r4/8/8/8/3RK3 w - - 0 1")
    static = _material_eval(free)
    got = qsearch(evaluate, [free], depth=2)[0]
    total += 1
    passed += got > static
    print(f"  {'ok  ' if got > static else 'FAIL'}  free rook is taken: "
          f"{static:+.3f} -> {got:+.3f}")

    # The trap the old version fell into: winning a pawn but losing the rook.
    # Declining has to win out.
    trap = chess.Board("4k3/8/2p5/3p4/8/8/8/3RK3 w - - 0 1")
    check("losing capture is declined",
          qsearch(evaluate, [trap], depth=3)[0], _material_eval(trap))

    # Being in check must not allow standing pat.
    checked = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
    got = qsearch(evaluate, [checked], depth=2)[0]
    print(f"  in check, cannot stand pat: {got:+.3f} "
          f"(static was {_material_eval(checked):+.3f})")

    # Mate found inside the search is genuine: the mated side has no move.
    mate_soon = chess.Board("7k/6pp/8/8/8/8/8/R6K w - - 0 1")
    got = qsearch(evaluate, [mate_soon], depth=2)[0]
    print(f"  rook lift toward a mating net: {got:+.3f}")

    calls[0] = 0
    batch = [chess.Board(), chess.Board("4k3/8/8/3r4/8/8/8/3RK3 w - - 0 1")]
    qsearch(evaluate, batch, depth=1)
    print(f"\n  {calls[0]} positions evaluated for a 2-board depth-1 search")

    print(f"\n{passed}/{total} checks passed")


if __name__ == "__main__":
    main()
