"""
search.py - play against the network with Monte Carlo tree search on top.

The plain policy net guesses a move from the position alone. This adds
lookahead: it plays moves out, evaluates the resulting positions with the
value head, and backs those judgements up the tree.

Usage:
    python search.py                        # 400 simulations per move
    python search.py --sims 1200            # stronger, slower
    python search.py --black
    python search.py --ckpt checkpoints/best.pt
    python search.py --contempt 0.4         # try harder to avoid draws
    python search.py --material 0.25        # more sceptical of sacrifices
    python search.py --cpuct 2.0            # narrower, deeper search

Commands during play: board, fen, undo, quit
"""

import math
import os
import sys

import torch
import torch.nn.functional as F
import chess

from train import ChessNet
from play import encode_board, parse_move, show

DEFAULT_CKPT = "checkpoints/best.pt"
DEFAULT_SIMS = 400
C_PUCT = 3.0

# How bad a draw is considered, from the perspective of the side to move.
# 0.0 treats a draw as neutral, which makes the search happy to repeat or
# shuffle in a winning position. A positive value makes it prefer to keep
# playing. Too high and it will avoid drawing even when a draw is the best
# available result.
DRAW_CONTEMPT = 0.3

# How much raw material counts toward a leaf evaluation, from 0 to 1.
# 0.0 trusts the value head completely. Higher values pull evaluations of
# material-down positions downward, so the search only likes a sacrifice
# when it finds concrete compensation rather than a promising-looking one.
MATERIAL_WEIGHT = 0.2
MATERIAL_SCALE = 3.0     # pawns of advantage that map to roughly 0.76

PIECE_VALUE = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.25,
               chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}


def material_balance(board):
    """Material from the side to move's perspective, in pawns."""
    total = 0.0
    for piece in board.piece_map().values():
        value = PIECE_VALUE[piece.piece_type]
        total += value if piece.color == board.turn else -value
    return total


class Node:
    __slots__ = ("prior", "children", "visits", "total_value")

    def __init__(self, prior=0.0):
        self.prior = prior
        self.children = None      # None means not expanded yet
        self.visits = 0
        self.total_value = 0.0

    @property
    def value(self):
        """Average outcome from the perspective of whoever moves at this node."""
        return self.total_value / self.visits if self.visits else 0.0


@torch.no_grad()
def expand(model, node, board):
    """Run the network, attach children with priors, return the value estimate."""
    logits, value = model(encode_board(board))
    logits = logits[0]

    # Collapse from/to duplicates (queen vs under-promotion) as in play.py.
    by_index = {}
    for move in board.legal_moves:
        idx = move.from_square * 64 + move.to_square
        if idx not in by_index or move.promotion == chess.QUEEN:
            by_index[idx] = move

    indices = list(by_index.keys())
    priors = F.softmax(logits[indices], dim=0)

    node.children = {
        by_index[i]: Node(prior=p.item())
        for i, p in zip(indices, priors)
    }
    return value.item()


def select_child(node, c_puct=C_PUCT):
    """PUCT: balance what looks good so far against what looks promising."""
    best_score = -float("inf")
    best = None
    sqrt_visits = math.sqrt(node.visits)

    for move, child in node.children.items():
        # child.value is from the child's perspective, so negate it.
        q = -child.value if child.visits else 0.0
        u = c_puct * child.prior * sqrt_visits / (1 + child.visits)
        score = q + u
        if score > best_score:
            best_score = score
            best = (move, child)

    return best


def terminal_value(board, contempt=DRAW_CONTEMPT):
    """Value for the side to move in a finished position."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        # Stalemate, repetition, 50-move and insufficient material all land
        # here. Scoring them slightly negative stops the search drifting into
        # a draw when it has a winning position and every move looks equally
        # good to a saturated value head.
        return -contempt
    return -1.0             # side to move has been checkmated


def run_search(model, root_board, sims, contempt=DRAW_CONTEMPT,
               material=MATERIAL_WEIGHT, c_puct=C_PUCT):
    root = Node()
    expand(model, root, root_board)

    for _ in range(sims):
        board = root_board.copy()
        node = root
        path = [node]

        # Walk down through already-expanded nodes.
        while node.children is not None and node.children:
            move, node = select_child(node, c_puct)
            board.push(move)
            path.append(node)
            if node.children is None:
                break

        value = terminal_value(board, contempt)
        if value is None:
            value = expand(model, node, board)
            # Blend in raw material. Terminal positions are left alone -
            # a mate is a mate regardless of who owns more pieces.
            if material:
                balance = math.tanh(material_balance(board) / MATERIAL_SCALE)
                value = (1.0 - material) * value + material * balance

        # Back the result up, flipping sign at every ply.
        for n in reversed(path):
            n.visits += 1
            n.total_value += value
            value = -value

    return root


def choose(root):
    """Most-visited move, plus the ranked list for display."""
    ranked = sorted(root.children.items(), key=lambda kv: kv[1].visits, reverse=True)
    return ranked[0][0], ranked


def main():
    ckpt_path = DEFAULT_CKPT
    if "--ckpt" in sys.argv:
        ckpt_path = sys.argv[sys.argv.index("--ckpt") + 1]

    sims = DEFAULT_SIMS
    if "--sims" in sys.argv:
        sims = int(sys.argv[sys.argv.index("--sims") + 1])

    contempt = DRAW_CONTEMPT
    if "--contempt" in sys.argv:
        contempt = float(sys.argv[sys.argv.index("--contempt") + 1])

    material = MATERIAL_WEIGHT
    if "--material" in sys.argv:
        material = float(sys.argv[sys.argv.index("--material") + 1])

    c_puct = C_PUCT
    if "--cpuct" in sys.argv:
        c_puct = float(sys.argv[sys.argv.index("--cpuct") + 1])

    human_is_white = "--black" not in sys.argv

    if not os.path.exists(ckpt_path):
        print(f"No checkpoint at {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ChessNet(blocks=ckpt.get("blocks", 6), channels=ckpt.get("channels", 128))
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"loaded {ckpt_path} - step {ckpt['step']:,}")
    print(f"{sims} simulations per move  |  c_puct {c_puct}  |  "
          f"contempt {contempt}  |  material {material}")
    print(f"you are {'White' if human_is_white else 'Black'}\n")

    board = chess.Board()
    human_colour = chess.WHITE if human_is_white else chess.BLACK
    show(board, human_is_white)

    while not board.is_game_over(claim_draw=True):
        if board.turn == human_colour:
            text = input("your move: ").strip()

            if text in ("quit", "exit"):
                return
            if text == "board":
                show(board, human_is_white)
                continue
            if text == "fen":
                print(board.fen())
                continue
            if text == "undo":
                if len(board.move_stack) >= 2:
                    board.pop()
                    board.pop()
                    show(board, human_is_white)
                continue

            move = parse_move(board, text)
            if move is None or move not in board.legal_moves:
                print("not a legal move")
                continue
            board.push(move)

        else:
            root = run_search(model, board, sims, contempt, material, c_puct)
            move, ranked = choose(root)
            san = board.san(move)

            summary = ", ".join(
                f"{board.san(m)} {n.visits}v {-n.value:+.2f}"
                for m, n in ranked[:3]
            )
            board.push(move)
            print(f"\nbot plays {san}")
            print(f"  searched: {summary}")

        show(board, human_is_white)

    print("game over:", board.result(claim_draw=True))


if __name__ == "__main__":
    main()
