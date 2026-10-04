"""
uci.py - speak the UCI protocol, so lichess-bot or any chess GUI can drive
this engine.

Run it directly to try it by hand:

    python uci.py
    uci
    isready
    position startpos moves e2e4
    go movetime 3000
    quit

Time management: given a clock, it estimates how many simulations fit in the
time budget from how fast recent searches ran, rather than using a fixed
count. Given "go nodes N" it runs exactly N simulations, which is what you
want for reproducible testing.
"""

import sys
import time

import torch
import chess

from train import ChessNet
from search_batched import (run_search, choose, C_PUCT, DRAW_CONTEMPT,
                            MATERIAL_WEIGHT, DEFAULT_BATCH, QUIESCE_PLIES,
                            POLICY_SMOOTHING, FORCED_VISIT_DEPTH,
                            SYZYGY_PATH)

ENGINE_NAME = "chessbot"
ENGINE_AUTHOR = "jimmy"

DEFAULT_CKPT = "checkpoints/evaltuned_best.pt"
DEFAULT_SIMS = 400
MIN_SIMS = 32
MAX_SIMS = 900000
MOVES_TO_GO = 30          # assume this many moves left when budgeting time


def out(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


class Engine:
    def __init__(self):
        self.model = None
        self.board = chess.Board()
        self.ckpt_path = DEFAULT_CKPT
        self.sims = DEFAULT_SIMS
        self.cpuct = C_PUCT
        self.contempt = DRAW_CONTEMPT
        self.material = MATERIAL_WEIGHT
        self.batch = DEFAULT_BATCH
        self.quiesce = QUIESCE_PLIES
        self.smoothing = POLICY_SMOOTHING
        self.forced_depth = FORCED_VISIT_DEPTH
        self.syzygy = SYZYGY_PATH
        self.trained_steps = 0
        self.eval_steps = 0
        self.blocks = 0
        self.channels = 0
        self.sims_per_second = 2000.0    # revised as we go

    def load(self):
        if self.model is not None:
            return
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        self.trained_steps = ckpt.get("step", 0)
        self.eval_steps = ckpt.get("eval_step", 0)
        self.blocks = ckpt.get("blocks", 6)
        self.channels = ckpt.get("channels", 128)
        model = ChessNet(blocks=ckpt.get("blocks", 6),
                         channels=ckpt.get("channels", 128))
        model.load_state_dict(ckpt["model"])
        model.eval()
        self.model = model

    # ------------------------------------------------------------ protocol

    def cmd_uci(self):
        # Load up front so the reported name can carry the actual version.
        # A GUI expects "uci" to answer quickly, but a couple of seconds is
        # well within tolerance and it makes the running build identifiable.
        try:
            self.load()
            version = (f"{self.blocks}x{self.channels}"
                       f"-s{self.trained_steps}"
                       f"-e{self.eval_steps}")
        except Exception:
            version = "unloaded"

        out(f"id name {ENGINE_NAME} {version}")
        out(f"id author {ENGINE_AUTHOR}")
        out(f"option name Checkpoint type string default {DEFAULT_CKPT}")
        out(f"option name Sims type spin default {DEFAULT_SIMS} "
            f"min {MIN_SIMS} max {MAX_SIMS}")
        out("option name CPuct type string default 2.0")
        out("option name Contempt type string default 0.3")
        out("option name Material type string default 0.2")
        out(f"option name Batch type spin default {DEFAULT_BATCH} min 1 max 512")
        out(f"option name Quiesce type spin default {QUIESCE_PLIES} min 0 max 32")
        out(f"option name Smooth type string default {POLICY_SMOOTHING}")
        out(f"option name Forced type spin default {FORCED_VISIT_DEPTH} min 0 max 4")
        out(f"option name Syzygy type string default {SYZYGY_PATH}")
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
                self.model = None            # reload on next search
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
                self.quiesce = max(0, min(32, int(value)))
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
        """Decide how many simulations this move gets."""
        params = {}
        for i, token in enumerate(tokens):
            if token in ("wtime", "btime", "winc", "binc", "movetime",
                         "nodes", "movestogo", "depth"):
                try:
                    params[token] = int(tokens[i + 1])
                except (IndexError, ValueError):
                    pass

        # Exact simulation count - used for reproducible testing.
        if "nodes" in params:
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
        sims = int(self.sims_per_second * seconds)
        return max(MIN_SIMS, min(MAX_SIMS, sims)), seconds

    def cmd_go(self, tokens):
        self.load()

        if self.board.is_game_over(claim_draw=True):
            out("bestmove 0000")
            return

        sims, _ = self.budget_sims(tokens)

        start = time.time()
        root = run_search(self.model, self.board, sims,
                          self.contempt, self.material, self.cpuct,
                          batch_size=self.batch, quiesce=self.quiesce,
                          syzygy=self.syzygy, smoothing=self.smoothing,
                          forced_depth=self.forced_depth)
        elapsed = max(time.time() - start, 1e-6)

        # Blend the observed rate into the estimate for next time.
        rate = sims / elapsed
        self.sims_per_second = 0.7 * self.sims_per_second + 0.3 * rate

        move, ranked = choose(root)
        score = -ranked[0][1].value
        out(f"info depth 1 nodes {sims} score cp {int(score * 400)} "
            f"pv {move.uci()}")
        out(f"bestmove {move.uci()}")


def main():
    engine = Engine()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        command = tokens[0]

        if command == "uci":
            engine.cmd_uci()
        elif command == "isready":
            engine.load()
            out("readyok")
        elif command == "setoption":
            engine.cmd_setoption(tokens)
        elif command == "ucinewgame":
            engine.board = chess.Board()
        elif command == "position":
            engine.cmd_position(tokens)
        elif command == "go":
            engine.cmd_go(tokens)
        elif command in ("stop", "ponderhit"):
            pass          # searches here are not interruptible
        elif command == "quit":
            break


if __name__ == "__main__":
    main()
