"""
play.py - play a game against your trained network.

Usage:
    python play.py                          # you are White, temperature 0.3
    python play.py --black                  # you are Black
    python play.py --temp 0                 # bot always plays its top move
    python play.py --temp 1.0               # bot plays loosely, more variety
    python play.py --ckpt checkpoints/snapshot.pt

Type moves as either SAN (Nf3, exd5, O-O) or UCI (g1f3, e4d5, e1g1).
Other commands: board, fen, undo, hint, quit
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import chess

from train import ChessNet, PLANES

DEFAULT_CKPT = "checkpoints/latest.pt"
DEFAULT_TEMP = 0.1


def encode_board(board):
    """Same 18-plane layout that train.py feeds the network."""
    x = np.zeros((1, PLANES, 64), dtype=np.float32)

    for square, piece in board.piece_map().items():
        plane = piece.piece_type - 1              # 0-5
        if not piece.color:                       # black
            plane += 6
        x[0, plane, square] = 1.0

    rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]
    for i, has in enumerate(rights):
        if has:
            x[0, 12 + i, :] = 1.0

    if board.turn == chess.WHITE:
        x[0, 16, :] = 1.0

    if board.ep_square is not None:
        x[0, 17, board.ep_square] = 1.0

    return torch.from_numpy(x.reshape(1, PLANES, 8, 8))


@torch.no_grad()
def think(model, board, temperature):
    """Return (chosen_move, ranked_list_of_(move, probability), value_estimate)."""
    logits, value = model(encode_board(board))
    logits = logits[0]

    legal = list(board.legal_moves)

    # Several legal moves can share a from/to pair (queen vs under-promotion).
    # The network does not distinguish them, so keep the queen promotion.
    by_index = {}
    for move in legal:
        idx = move.from_square * 64 + move.to_square
        if idx not in by_index or move.promotion == chess.QUEEN:
            by_index[idx] = move

    indices = list(by_index.keys())
    scores = logits[indices]

    if temperature <= 0:
        probs = torch.zeros_like(scores)
        probs[scores.argmax()] = 1.0
    else:
        probs = F.softmax(scores / temperature, dim=0)

    ranked = sorted(
        zip((by_index[i] for i in indices), probs.tolist()),
        key=lambda pair: pair[1],
        reverse=True,
    )

    if temperature <= 0:
        chosen = ranked[0][0]
    else:
        pick = torch.multinomial(probs, 1).item()
        chosen = by_index[indices[pick]]

    return chosen, ranked, value.item()


def parse_move(board, text):
    """Accept SAN or UCI, return a Move or None."""
    for parser in (board.parse_san, board.parse_uci):
        try:
            return parser(text)
        except ValueError:
            continue
    return None


def show(board, human_is_white):
    print()
    print(board if human_is_white else board.mirror().transform(chess.flip_vertical))
    print()


def main():
    ckpt_path = DEFAULT_CKPT
    if "--ckpt" in sys.argv:
        ckpt_path = sys.argv[sys.argv.index("--ckpt") + 1]

    temperature = DEFAULT_TEMP
    if "--temp" in sys.argv:
        temperature = float(sys.argv[sys.argv.index("--temp") + 1])

    human_is_white = "--black" not in sys.argv

    if not os.path.exists(ckpt_path):
        print(f"No checkpoint at {ckpt_path} - has training saved yet?")
        sys.exit(1)

    # CPU on purpose: one position at a time is trivial work, and this leaves
    # the GPU free for the training run you probably still have going.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ChessNet(blocks=ckpt.get("blocks", 6), channels=ckpt.get("channels", 128))
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"loaded {ckpt_path} - trained to step {ckpt['step']:,}")
    print(f"temperature {temperature}  |  you are "
          f"{'White' if human_is_white else 'Black'}")
    print("commands: board, fen, undo, hint, quit\n")

    board = chess.Board()
    human_colour = chess.WHITE if human_is_white else chess.BLACK
    show(board, human_is_white)

    while not board.is_game_over():
        if board.turn == human_colour:
            text = input("your move: ").strip()

            if text in ("quit", "exit"):
                print("game abandoned")
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
                else:
                    print("nothing to undo")
                continue
            if text == "hint":
                _, ranked, value = think(model, board, temperature)
                print("  network likes:", ", ".join(
                    f"{board.san(m)} {p*100:.0f}%" for m, p in ranked[:3]))
                continue

            move = parse_move(board, text)
            if move is None or move not in board.legal_moves:
                print("not a legal move - try SAN like Nf3 or UCI like g1f3")
                continue
            board.push(move)

        else:
            move, ranked, value = think(model, board, temperature)
            san = board.san(move)
            board.push(move)
            print(f"\nbot plays {san}")
            print("  considered:", ", ".join(
                f"{m.uci()} {p*100:.0f}%" for m, p in ranked[:3]))
            print(f"  position score for the bot: {value:+.2f}")

        show(board, human_is_white)

    print("game over:", board.result(), "-", board.outcome().termination.name)


if __name__ == "__main__":
    main()
