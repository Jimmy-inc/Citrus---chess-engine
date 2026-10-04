"""Lazy, reusable Stockfish 18 process for the Play tab's eval bar.

Started on first use (not at app launch) and kept running for reuse -
starting the engine takes a few hundred ms, too slow to pay on every move.
"""

import atexit
import shutil
import threading

STOCKFISH_PATH = shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"
ANALYSIS_TIME = 0.3  # seconds - "lite": fast enough for a live eval bar

_engine = None
_lock = threading.Lock()


def available():
    return shutil.which("stockfish") is not None or STOCKFISH_PATH is not None


def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            import chess.engine

            _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            atexit.register(_close)
        return _engine


def _close():
    global _engine
    if _engine is not None:
        try:
            _engine.quit()
        except Exception:
            pass
        _engine = None


def white_cp(board):
    """White-perspective centipawn score for the position, or None if
    Stockfish isn't available or the position is already decided."""
    import chess.engine

    engine = _get_engine()
    with _lock:
        info = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))
    score = info["score"].white()
    return score.score(mate_score=10000)
