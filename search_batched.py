"""
search_batched.py - the same MCTS as search.py, but it gathers a batch of
leaf positions and evaluates them in a single forward pass on the GPU.

Drop-in replacement: run_search() and choose() take the same arguments as
the ones in search.py, so match.py and uci.py only need their import line
changed.

Play against it:

    python search_batched.py --sims 2000
    python search_batched.py --sims 2000 --quiesce 8    # resolve captures first
    python search_batched.py --sims 2000 --syzygy syzygy  # perfect endgames
    python search_batched.py --sims 2000 --smooth 0.03     # never ignore a move
    python search_batched.py --sims 4000 --forced 2        # check every reply
    python search_batched.py --sims 4000 --black --cpuct 2

Benchmark it against the old search:

    python search_batched.py --bench --sims 2000

Why this is faster
------------------
search.py evaluates one position per network call. A batch of one barely
touches the GPU - almost all the time goes on launch overhead rather than
arithmetic, which is why search.py runs on CPU at all. Evaluating 64
positions at once costs little more than evaluating one, so the simulation
rate climbs sharply.

The catch is that MCTS is sequential: every descent depends on statistics
that earlier simulations produced. Collecting 64 leaves naively would return
64 copies of the same position. Virtual loss solves this - on the way down,
each node is temporarily marked as though a simulation had just gone there
and lost, which pushes the next descent elsewhere. The marks are removed
when the real results are backed up.
"""

import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import chess

from train import ChessNet
from play import encode_board, parse_move, show
from quiescence import qsearch
from endgame import probe as probe_tablebase

DEFAULT_CKPT = "checkpoints/evaltuned_best.pt"
DEFAULT_SIMS = 400
DEFAULT_BATCH = 48

C_PUCT = 2.0
DRAW_CONTEMPT = 0.3
MATERIAL_WEIGHT = 0.2
MATERIAL_SCALE = 3.0

# How heavily a pending visit is penalised while the batch is being gathered.
# Larger values spread the batch across more of the tree but distort the
# statistics more; 1.0 is the usual choice.
VIRTUAL_LOSS = 1.0

# Captures are played out before the network sees a position, so it judges a
# settled board rather than one with pieces hanging mid-exchange. 0 disables
# it. Costs Python time per leaf, so it is a trade, not a free win.
QUIESCE_PLIES = 0

# Folder holding Syzygy tablebases. When a leaf has few enough pieces the
# result is looked up rather than guessed, which is worth far more than any
# amount of evaluation training in those positions. Empty string disables it.
SYZYGY_PATH = "syzygy"

# Blend a little uniform probability into every policy output. The
# exploration bonus is multiplied by the prior, so a move the net rates at
# essentially zero can never be tried, no matter how many simulations run -
# and if that move happens to be the refutation, the search will never find
# out. A floor of a few percent costs almost nothing and removes the blind
# spot. 0 disables it.
POLICY_SMOOTHING = 0.0

# Depth up to which every child must be visited at least once before PUCT
# takes over. 1 covers your own candidate moves, 2 covers every reply to
# them - which is what catches quiet one-move refutations, at the cost of
# roughly (moves x replies) simulations before the real search begins.
FORCED_VISIT_DEPTH = 0

PIECE_VALUE = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.25,
               chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}


class Node:
    __slots__ = ("prior", "children", "visits", "total_value")

    def __init__(self, prior=0.0):
        self.prior = prior
        self.children = None
        self.visits = 0
        self.total_value = 0.0

    @property
    def value(self):
        return self.total_value / self.visits if self.visits else 0.0


# ---------------------------------------------------------------- helpers


def material_balance(board):
    total = 0.0
    for piece in board.piece_map().values():
        value = PIECE_VALUE[piece.piece_type]
        total += value if piece.color == board.turn else -value
    return total


def terminal_value(board, contempt=DRAW_CONTEMPT):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return -contempt
    return -1.0


def select_unvisited(node):
    """
    The first child nothing has looked at yet, or None.

    A move the policy head rates near zero gets an exploration bonus near
    zero too, so it can go unvisited however long the search runs - and if
    it happens to be the refutation, the search never finds out. Forcing one
    visit is enough: once the value comes back bad, PUCT keeps returning
    there on its own.
    """
    for move, child in node.children.items():
        if child.visits == 0:
            return move, child
    return None


def select_child(node, c_puct):
    best_score = -float("inf")
    best = None
    sqrt_visits = math.sqrt(node.visits)
    for move, child in node.children.items():
        q = -child.value if child.visits else 0.0
        u = c_puct * child.prior * sqrt_visits / (1 + child.visits)
        score = q + u
        if score > best_score:
            best_score = score
            best = (move, child)
    return best


def legal_index_map(board):
    """Collapse from/to duplicates, keeping the queen promotion."""
    by_index = {}
    for move in board.legal_moves:
        idx = move.from_square * 64 + move.to_square
        if idx not in by_index or move.promotion == chess.QUEEN:
            by_index[idx] = move
    return by_index


def attach_children(node, board, logits, smoothing=POLICY_SMOOTHING):
    """Give a leaf its children, with priors from an already-computed row."""
    by_index = legal_index_map(board)
    indices = list(by_index.keys())
    priors = F.softmax(logits[indices], dim=0)

    if smoothing:
        uniform = 1.0 / len(indices)
        priors = (1.0 - smoothing) * priors + smoothing * uniform

    node.children = {by_index[i]: Node(prior=float(p))
                     for i, p in zip(indices, priors)}


def add_virtual_loss(path):
    # path[0] is the root, which is never selected against.
    for node in path[1:]:
        node.visits += VIRTUAL_LOSS
        # total_value is from this node's own perspective, and the parent
        # scores it as -value. Pushing it up makes the parent like it less.
        node.total_value += VIRTUAL_LOSS


def remove_virtual_loss(path):
    for node in path[1:]:
        node.visits -= VIRTUAL_LOSS
        node.total_value -= VIRTUAL_LOSS


def backup(path, value):
    for node in reversed(path):
        node.visits += 1
        node.total_value += value
        value = -value


# ---------------------------------------------------------------- search


_device_cache = {}


def device_for(model):
    """Move the model onto the GPU once, and remember where it went."""
    key = id(model)
    if key not in _device_cache:
        from train import pick_device
        device = pick_device()
        model.to(device)
        _device_cache[key] = device
    return _device_cache[key]


@torch.no_grad()
def evaluate_batch(model, boards, device):
    x = torch.cat([encode_board(b) for b in boards], dim=0).to(device)
    logits, values = model(x)
    return logits.cpu(), values.cpu()


def leaf_value(net_value, board, material):
    """The network's opinion, optionally pulled toward raw material."""
    if not material:
        return net_value
    balance = math.tanh(material_balance(board) / MATERIAL_SCALE)
    return (1.0 - material) * net_value + material * balance


def make_evaluator(model, device, material, syzygy):
    """
    The callback the quiescence search uses for a whole level at a time.

    Tablebase hits are answered exactly and never reach the network, which
    also keeps them out of the batch.
    """
    @torch.no_grad()
    def evaluate(boards):
        values = [None] * len(boards)
        needed = []
        needed_index = []

        for i, board in enumerate(boards):
            exact = probe_tablebase(board, 0.0, syzygy) if syzygy else None
            if exact is not None:
                values[i] = exact
            else:
                needed.append(board)
                needed_index.append(i)

        if needed:
            _, raw = evaluate_batch(model, needed, device)
            for k, i in enumerate(needed_index):
                values[i] = leaf_value(raw[k].item(), needed[k], material)

        return values

    return evaluate


@torch.no_grad()
def run_search(model, root_board, sims, contempt=DRAW_CONTEMPT,
               material=MATERIAL_WEIGHT, c_puct=C_PUCT,
               batch_size=DEFAULT_BATCH, quiesce=QUIESCE_PLIES,
               syzygy=SYZYGY_PATH, smoothing=POLICY_SMOOTHING,
               forced_depth=FORCED_VISIT_DEPTH):
    device = device_for(model)
    root = Node()

    # The root has to be expanded on its own before anything can descend.
    root_logits, _ = evaluate_batch(model, [root_board], device)
    attach_children(root, root_board, root_logits[0], smoothing)

    done = 0
    while done < sims:
        want = min(batch_size, sims - done)
        pending = []          # (node, board, path)
        seen = set()
        attempts = 0

        while len(pending) < want and attempts < want * 3:
            attempts += 1
            board = root_board.copy()
            node = root
            path = [node]

            depth = 0
            while node.children:
                picked = None
                if depth < forced_depth:
                    picked = select_unvisited(node)
                if picked is None:
                    picked = select_child(node, c_puct)
                move, child = picked
                board.push(move)
                node = child
                path.append(node)
                depth += 1

            add_virtual_loss(path)

            value = terminal_value(board, contempt)
            if value is None and syzygy:
                # Solved position: use the true result rather than the net's
                # opinion of it.
                value = probe_tablebase(board, contempt, syzygy)
            if value is not None:
                # No network call needed - the position is already decided.
                remove_virtual_loss(path)
                backup(path, value)
                done += 1
                continue

            if id(node) in seen:
                # Another descent in this batch already claimed this leaf.
                remove_virtual_loss(path)
                continue

            seen.add(id(node))
            pending.append((node, board, path))

        if not pending:
            if attempts >= want * 3:
                break          # tree is saturated; nothing new to visit
            continue

        boards = [board for _, board, _ in pending]
        logits, raw = evaluate_batch(model, boards, device)

        # The priors come from this level's own network call. The values may
        # get refined below, but the leaf evaluation is the starting point.
        base = [leaf_value(raw[i].item(), boards[i], material)
                for i in range(len(pending))]

        if quiesce:
            values = qsearch(make_evaluator(model, device, material, syzygy),
                             boards, quiesce, contempt, stand_pat=base)
        else:
            values = base

        for i, (node, board, path) in enumerate(pending):
            attach_children(node, board, logits[i], smoothing)
            remove_virtual_loss(path)
            backup(path, values[i])
            done += 1

    return root


def choose(root):
    ranked = sorted(root.children.items(), key=lambda kv: kv[1].visits,
                    reverse=True)
    return ranked[0][0], ranked


# ---------------------------------------------------------------- bench


def bench():
    ckpt_path = DEFAULT_CKPT
    if "--ckpt" in sys.argv:
        ckpt_path = sys.argv[sys.argv.index("--ckpt") + 1]
    sims = int(sys.argv[sys.argv.index("--sims") + 1]) if "--sims" in sys.argv else 2000
    batch = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else DEFAULT_BATCH
    quiesce = int(sys.argv[sys.argv.index("--quiesce") + 1]) if "--quiesce" in sys.argv else QUIESCE_PLIES

    if not os.path.exists(ckpt_path):
        print(f"no checkpoint at {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    def fresh():
        m = ChessNet(blocks=ckpt.get("blocks", 6),
                     channels=ckpt.get("channels", 128))
        m.load_state_dict(ckpt["model"])
        m.eval()
        return m

    # A middlegame position gives a fairer picture than the start position.
    board = chess.Board(
        "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9")

    import search as old

    model_old = fresh()
    start = time.time()
    old.run_search(model_old, board, sims, DRAW_CONTEMPT, MATERIAL_WEIGHT, C_PUCT)
    old_elapsed = time.time() - start

    model_new = fresh()
    device = device_for(model_new)
    # One throwaway search so Metal finishes compiling its kernels.
    run_search(model_new, board, 200, batch_size=batch, quiesce=quiesce)
    start = time.time()
    root = run_search(model_new, board, sims, DRAW_CONTEMPT, MATERIAL_WEIGHT,
                      C_PUCT, batch_size=batch, quiesce=quiesce)
    new_elapsed = time.time() - start

    print(f"\nposition: quiet middlegame, {sims:,} simulations")
    print(f"  search.py         : {old_elapsed:6.2f}s   "
          f"{sims/old_elapsed:7.0f} sims/s")
    print(f"  search_batched.py : {new_elapsed:6.2f}s   "
          f"{sims/new_elapsed:7.0f} sims/s   "
          f"on {device}, batch {batch}, quiesce {quiesce}")
    print(f"  speedup           : {old_elapsed/new_elapsed:.1f}x")

    move, ranked = choose(root)
    print(f"\n  batched search picks {board.san(move)}, "
          f"top three: " + ", ".join(
              f"{board.san(m)} {n.visits}v" for m, n in ranked[:3]))


# ---------------------------------------------------------------- play


def arg(name, default, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def play():
    ckpt_path = arg("--ckpt", DEFAULT_CKPT)
    sims = arg("--sims", DEFAULT_SIMS, int)
    batch = arg("--batch", DEFAULT_BATCH, int)
    contempt = arg("--contempt", DRAW_CONTEMPT, float)
    material = arg("--material", MATERIAL_WEIGHT, float)
    c_puct = arg("--cpuct", C_PUCT, float)
    quiesce = arg("--quiesce", QUIESCE_PLIES, int)
    syzygy = arg("--syzygy", SYZYGY_PATH)
    smoothing = arg("--smooth", POLICY_SMOOTHING, float)
    forced_depth = arg("--forced", FORCED_VISIT_DEPTH, int)
    human_is_white = "--black" not in sys.argv

    if not os.path.exists(ckpt_path):
        print(f"no checkpoint at {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ChessNet(blocks=ckpt.get("blocks", 6),
                     channels=ckpt.get("channels", 128))
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = device_for(model)

    print(f"loaded {ckpt_path} - step {ckpt.get('step', 0):,}")
    print(f"{sims} simulations per move on {device}, batch {batch}")
    print(f"c_puct {c_puct}  |  contempt {contempt}  |  material {material}"
          f"  |  quiesce {quiesce}")
    print(f"policy smoothing {smoothing}  |  forced visits to depth "
          f"{forced_depth}")
    from endgame import available as _tb_available
    print(f"tablebase: {'in use at ./' + syzygy if syzygy and _tb_available(syzygy) else 'none'}")
    print(f"you are {'White' if human_is_white else 'Black'}")
    print("commands: board, fen, undo, quit\n")

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
            start = time.time()
            root = run_search(model, board, sims, contempt, material,
                              c_puct, batch_size=batch, quiesce=quiesce,
                              syzygy=syzygy, smoothing=smoothing,
                              forced_depth=forced_depth)
            elapsed = time.time() - start
            move, ranked = choose(root)
            san = board.san(move)

            summary = ", ".join(
                f"{board.san(m)} {int(n.visits)}v {-n.value:+.2f}"
                for m, n in ranked[:3])
            board.push(move)
            print(f"\nbot plays {san}   ({elapsed:.1f}s, "
                  f"{sims/elapsed:.0f} sims/s)")
            print(f"  searched: {summary}")

        show(board, human_is_white)

    print("game over:", board.result(claim_draw=True))


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        play()
