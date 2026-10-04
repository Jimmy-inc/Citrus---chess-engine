"""
uci_v2.py - speak the UCI protocol on behalf of the version two net.

Same job as uci.py, but driving search_v2 so it works with the canonical
encoding and the convolutional policy head. The two can coexist: point
lichess-bot at whichever launcher corresponds to the net you want to run.

Try it by hand:

    python uci_v2.py
    uci
    isready
    position startpos moves e2e4
    go movetime 3000
    quit

Time management estimates how many simulations fit the budget from how fast
recent searches actually ran, rather than assuming a rate. Given "go nodes N"
it runs exactly N simulations, which is what you want for reproducible tests.
"""

import sys
import time

import torch
import chess

import search_v2 as engine

ENGINE_NAME = "chessbot-v2"
ENGINE_AUTHOR = "jimmy"

DEFAULT_CKPT = "checkpoints/v2_best.pt"
DEFAULT_SIMS = 800
MIN_SIMS = 32
MAX_SIMS = 300_000
MOVES_TO_GO = 30          # assumed moves remaining when budgeting the clock


def out(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


class Engine:
    def __init__(self):
        self.model = None
        self.board = chess.Board()
        self.ckpt_path = DEFAULT_CKPT
        self.sims = DEFAULT_SIMS
        self.cpuct = engine.C_PUCT
        self.contempt = engine.DRAW_CONTEMPT
        self.material = engine.MATERIAL_WEIGHT
        self.batch = engine.DEFAULT_BATCH
        self.quiesce = engine.QUIESCE_DEPTH
        self.smoothing = engine.POLICY_SMOOTHING
        self.forced_depth = engine.FORCED_VISIT_DEPTH
        self.syzygy = engine.SYZYGY_PATH

        self.blocks = 0
        self.channels = 0
        self.step = 0
        self.sims_per_second = 1500.0     # revised from measured searches

    def load(self):
        if self.model is not None:
            return
        model, ckpt = engine.load(self.ckpt_path)
        self.model = model
        self.blocks = ckpt.get("blocks", 12)
        self.channels = ckpt.get("channels", 256)
        self.step = ckpt.get("step", 0)
        out(f"info string loaded checkpoint {self.ckpt_path} step {self.step}")

    # ------------------------------------------------------------ protocol

    def cmd_uci(self):
        # Load first so the reported name can carry the real version. A GUI
        # tolerates a couple of seconds here, and it makes the running build
        # identifiable from the log.
        try:
            self.load()
            version = f"{self.blocks}x{self.channels}-s{self.step}"
        except Exception:
            version = "unloaded"

        out(f"id name {ENGINE_NAME} {version}")
        out(f"id author {ENGINE_AUTHOR}")
        out(f"option name Checkpoint type string default {DEFAULT_CKPT}")
        out(f"option name Sims type spin default {DEFAULT_SIMS} "
            f"min {MIN_SIMS} max {MAX_SIMS}")
        out(f"option name CPuct type string default {engine.C_PUCT}")
        out(f"option name Contempt type string default {engine.DRAW_CONTEMPT}")
        out(f"option name Material type string default {engine.MATERIAL_WEIGHT}")
        out(f"option name Batch type spin default {engine.DEFAULT_BATCH} "
            f"min 1 max 512")
        out(f"option name Quiesce type spin default {engine.QUIESCE_DEPTH} "
            f"min 0 max 8")
        out(f"option name Smooth type string default {engine.POLICY_SMOOTHING}")
        out(f"option name Forced type spin default "
            f"{engine.FORCED_VISIT_DEPTH} min 0 max 4")
        out(f"option name Syzygy type string default {engine.SYZYGY_PATH}")
        out("uciok")

    def cmd_setoption(self, tokens):
        if "name" not in tokens:
            return
        name_at = tokens.index("name") + 1
        value_at = tokens.index("value") + 1 if "value" in tokens else None
        name = " ".join(tokens[name_at:(value_at - 1 if value_at else None)])
        value = " ".join(tokens[value_at:]) if value_at else ""

        key = name.strip().lower()
        try:
            if key == "checkpoint":
                self.ckpt_path = value.strip()
                self.model = None            # reload on the next search
            elif key == "sims":
                self.sims = max(MIN_SIMS, min(MAX_SIMS, int(value)))
            elif key == "cpuct":
                self.cpuct = float(value)
            elif key == "contempt":
                self.contempt = float(value)
            elif key == "material":
                self.material = float(value)
            elif key == "batch":
                self.batch = max(1, min(512, int(value)))
            elif key == "quiesce":
                self.quiesce = max(0, min(8, int(value)))
            elif key == "smooth":
                self.smoothing = max(0.0, min(1.0, float(value)))
            elif key == "forced":
                self.forced_depth = max(0, min(4, int(value)))
            elif key == "syzygy":
                self.syzygy = value.strip()
        except ValueError:
            pass

    def cmd_position(self, tokens):
        if len(tokens) < 2:
            return

        if tokens[1] == "startpos":
            self.board = chess.Board()
            rest = tokens[2:]
        elif tokens[1] == "fen":
            fen_parts = []
            i = 2
            while i < len(tokens) and tokens[i] != "moves":
                fen_parts.append(tokens[i])
                i += 1
            self.board = chess.Board(" ".join(fen_parts))
            rest = tokens[i:]
        else:
            return

        if rest and rest[0] == "moves":
            for uci in rest[1:]:
                try:
                    self.board.push(chess.Move.from_uci(uci))
                except ValueError:
                    break

    def budget_sims(self, tokens):
        """
        Returns (simulation cap, seconds) for this move.

        When a clock is involved the seconds are what actually governs -
        the cap is set generously above the estimate so the deadline is
        what stops the search, not an underestimated rate.
        """
        params = {}
        for i, token in enumerate(tokens):
            if token in ("wtime", "btime", "winc", "binc", "movetime",
                         "nodes", "movestogo", "depth"):
                try:
                    params[token] = int(tokens[i + 1])
                except (IndexError, ValueError):
                    pass

        if "nodes" in params:
            # An explicit node count means exactly that, no deadline.
            return max(MIN_SIMS, min(MAX_SIMS, params["nodes"])), None

        if "movetime" in params:
            seconds = params["movetime"] / 1000.0
        elif "wtime" in params or "btime" in params:
            ours = params.get("wtime" if self.board.turn == chess.WHITE
                              else "btime", 60_000) / 1000.0
            inc = params.get("winc" if self.board.turn == chess.WHITE
                             else "binc", 0) / 1000.0
            togo = params.get("movestogo", MOVES_TO_GO) or MOVES_TO_GO
            seconds = ours / togo + inc * 0.8
            seconds = min(seconds, ours * 0.3)     # never gamble the clock
        else:
            return self.sims, None

        seconds = max(0.05, seconds * 0.85)        # leave overhead room
        # Head room of 3x over the estimate, so a low estimate cannot cut
        # the search short - the deadline ends it instead.
        cap = int(self.sims_per_second * seconds * 3)
        return max(MIN_SIMS, min(MAX_SIMS, cap)), seconds

    def cmd_go(self, tokens):
        self.load()

        if self.board.is_game_over(claim_draw=True):
            out("bestmove 0000")
            return

        sims, seconds = self.budget_sims(tokens)

        start = time.time()
        deadline = start + seconds if seconds else None
        root = engine.run_search(self.model, self.board, sims,
                                 self.contempt, self.material, self.cpuct,
                                 batch_size=self.batch, quiesce=self.quiesce,
                                 syzygy=self.syzygy, smoothing=self.smoothing,
                                 forced_depth=self.forced_depth,
                                 deadline=deadline)
        elapsed = max(time.time() - start, 1e-6)
        actual = engine.visit_count(root)

        # Fold the measured rate into the estimate for the next move.
        self.sims_per_second = (0.7 * self.sims_per_second
                                + 0.3 * (actual / elapsed))

        move, ranked = engine.choose(root)
        score = -ranked[0][1].value
        # nodes and nps are the real counts, so the log shows whether the
        # budget was actually spent rather than what was planned.
        out(f"info depth 1 nodes {actual} nps {int(actual/elapsed)} "
            f"time {int(elapsed*1000)} score cp {int(score * 400)} "
            f"pv {move.uci()}")
        out(f"bestmove {move.uci()}")


def main():
    state = Engine()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        command = tokens[0]

        if command == "uci":
            state.cmd_uci()
        elif command == "isready":
            state.load()
            out("readyok")
        elif command == "setoption":
            state.cmd_setoption(tokens)
        elif command == "ucinewgame":
            state.board = chess.Board()
        elif command == "position":
            state.cmd_position(tokens)
        elif command == "go":
            state.cmd_go(tokens)
        elif command in ("stop", "ponderhit"):
            pass          # searches here are not interruptible
        elif command == "quit":
            break


if __name__ == "__main__":
    main()
