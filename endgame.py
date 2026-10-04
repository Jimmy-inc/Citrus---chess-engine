"""
endgame.py - perfect play in simple endgames, via Syzygy tablebases.

Your net has never learned endgame technique properly. The Lichess games it
started on contained real endgames, but that training is buried under half a
million steps of everything since - and the TCEC games were adjudicated
before endgames were reached, while the evaluation database is mostly
middlegame and tactical positions people submitted for analysis.

Tablebases sidestep the problem entirely. For positions with few enough
pieces the result is a solved fact, not an estimate, so there is nothing to
learn and nothing to get wrong.

Download the 3-4-5 piece tables (about 1GB) into a syzygy/ folder:

    mkdir -p ~/Desktop/chessbot/syzygy
    cd ~/Desktop/chessbot/syzygy
    curl -O https://tablebase.lichess.ovh/tables/standard/3-4-5/KQvK.rtbw
    ...

or grab the whole 3-4-5 directory with a recursive download. 6-piece tables
exist too but run to roughly 150GB, which is not worth it here.

Self-test:
    python endgame.py
"""

import os

import chess
import chess.syzygy

DEFAULT_PATH = "syzygy"
MAX_PIECES = 5

_tablebase = None
_tried_to_open = False


def open_tablebase(path=DEFAULT_PATH):
    """Open the tablebase once and hold it. Returns None if unavailable."""
    global _tablebase, _tried_to_open

    if _tried_to_open:
        return _tablebase
    _tried_to_open = True

    if not os.path.isdir(path):
        return None
    try:
        _tablebase = chess.syzygy.open_tablebase(path)
    except (OSError, ValueError):
        _tablebase = None
    return _tablebase


def available(path=DEFAULT_PATH):
    return open_tablebase(path) is not None


def probe(board, contempt=0.0, path=DEFAULT_PATH, max_pieces=MAX_PIECES):
    """
    The true value of this position from the side to move's perspective,
    or None if the tablebase cannot answer.

    Syzygy needs castling rights gone - a position where either side can
    still castle is not in the tables.
    """
    tablebase = open_tablebase(path)
    if tablebase is None:
        return None
    if board.castling_rights:
        return None
    if chess.popcount(board.occupied) > max_pieces:
        return None

    try:
        wdl = tablebase.probe_wdl(board)
    except (chess.syzygy.MissingTableError, KeyError, IndexError, ValueError):
        return None

    if wdl > 0:
        return 1.0            # win, including cursed wins
    if wdl < 0:
        return -1.0           # loss, including blessed losses
    return -contempt          # a real draw, scored like any other


# ---------------------------------------------------------------- self-test


def main():
    path = DEFAULT_PATH
    tablebase = open_tablebase(path)
    if tablebase is None:
        print(f"no tablebase found at ./{path}/")
        print("see the notes at the top of this file for how to fetch one")
        return

    cases = [
        ("KQ vs K, white to move", "8/8/8/4k3/8/8/8/3QK3 w - - 0 1", 1.0),
        ("KQ vs K, black to move", "8/8/8/4k3/8/8/8/3QK3 b - - 0 1", -1.0),
        ("bare kings", "8/8/8/4k3/8/8/8/4K3 w - - 0 1", 0.0),
        ("K+P vs K, pawn on the 7th", "k7/7P/8/8/8/8/8/6K1 w - - 0 1", None),
    ]

    passed = 0
    for name, fen, expect in cases:
        board = chess.Board(fen)
        value = probe(board)
        if expect is None:
            ok = value is not None
            detail = "should be answerable"
        else:
            ok = value == expect
            detail = f"expected {expect:+.1f}"
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}: got "
              f"{'None' if value is None else f'{value:+.1f}'}  ({detail})")
        passed += ok

    print(f"\n{passed}/{len(cases)} checks passed")


if __name__ == "__main__":
    main()
