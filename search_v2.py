"""
search_v2.py - batched MCTS for the version two network.

The search itself is unchanged from search_batched.py: PUCT selection,
virtual loss so a batch of leaves can be gathered at once, quiescence, and
tablebase probes. Only the encoding differs.

What changed, and why it matters:

  * positions are encoded canonically, so the same code path serves both
    colours and the net never sees a mirrored problem;
  * the policy is 4672 move-type outputs rather than 4096 from/to pairs,
    which means underpromotions now have their own entries. The old search
    had to collapse a knight promotion onto the queen promotion's slot and
    hope; this one can tell them apart.

Play against it:
    python search_v2.py --sims 2000
    python search_v2.py --sims 4000 --black --quiesce 2

Benchmark it:
    python search_v2.py --bench --sims 2000
"""

import itertools
import math
import os
import struct
import sys
import time
from array import array

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval
import chess

from train_v2 import ChessNetV2, pick_device
import encoding_v2
from encoding_v2 import encode_boards, policy_indices
from quiescence import qsearch
from endgame import probe as probe_tablebase

DEFAULT_CKPT = "checkpoints/v2_best.pt"
DEFAULT_SIMS = 800
DEFAULT_BATCH = 48

C_PUCT = 2.0
DRAW_CONTEMPT = 0.0
MATERIAL_WEIGHT = 0.0
MATERIAL_SCALE = 3.0
QUIESCE_DEPTH = 0
POLICY_SMOOTHING = 0.0
# First-play urgency: what a move nobody has searched yet is assumed to be
# worth. None is the original rule, "dead equal" (0). A number r assumes it
# is a little worse than this position has looked so far:
#   parent value - r * sqrt(prior share of the moves already searched)
# Leela's default is around 0.3. Same scale as values, -1 to 1.
FPU_REDUCTION = None
FORCED_VISIT_DEPTH = 0
SYZYGY_PATH = "syzygy"

# True restores the old draw rule (claimable counts as drawn), so the two can
# be matched against each other with match_v2.py --claim-draws.
CLAIM_DRAWS = False

# Where legal-move priors are softmaxed. On CUDA the device does it: launching
# a few small kernels is cheap and the CPU is the bottleneck. On Apple's MPS
# each launch costs ~40us, so the CPU does it. None means choose by device;
# True or False forces one, for benchmarking.
PRIORS_ON_DEVICE = None

# However tight the clock, never play from a root this thinly searched.
MIN_SIMS_BEFORE_DEADLINE = 32

VIRTUAL_LOSS = 1.0

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
    """Material from the side to move's perspective, in pawns."""
    total = 0.0
    for piece in board.piece_map().values():
        value = PIECE_VALUE[piece.piece_type]
        total += value if piece.color == board.turn else -value
    return total


def terminal_value(board, contempt=DRAW_CONTEMPT):
    # A draw counts once it has happened, not when some move could claim one:
    # that test tries every legal move at every leaf, and calls a position
    # drawn even when the side able to claim is winning and never would.
    if CLAIM_DRAWS:
        outcome = board.outcome(claim_draw=True)
    else:
        outcome = board.outcome()
        if outcome is None and (board.halfmove_clock >= 100
                                or board.is_repetition(3)):
            return -contempt
    if outcome is None:
        return None
    if outcome.winner is None:
        return -contempt
    return -1.0


def leaf_terminal(board, moves, contempt):
    """
    terminal_value for a leaf whose legal moves are already generated.

    Gives exactly the same answers under the current draw rule, but reuses
    the move list the children are built from instead of generating moves
    again to look for checkmate and stalemate.
    """
    if not moves:
        return -1.0 if board.is_check() else -contempt
    if (board.is_insufficient_material() or board.halfmove_clock >= 100
            or board.is_repetition(3)):
        return -contempt
    return None


def select_unvisited(node):
    """The first child nothing has looked at yet, or None."""
    for move, child in node.children.items():
        if child.visits == 0:
            return move, child
    return None


def select_child(node, c_puct):
    # The hottest loop in the program: Node.value is inlined, and the
    # arithmetic keeps the original order so scores stay bit-identical.
    best_score = -math.inf
    best = None
    sqrt_visits = math.sqrt(node.visits)
    for item in node.children.items():
        child = item[1]
        n = child.visits
        score = ((-(child.total_value / n) if n else 0.0)
                 + c_puct * child.prior * sqrt_visits / (1 + n))
        if score > best_score:
            best_score = score
            best = item
    return best


def select_child_fpu(node, c_puct, reduction):
    """select_child, but unsearched moves start at the FPU estimate.

    A node's total_value is from the side to move there - the side choosing
    now - so its average is this position's value from the chooser's view,
    the same view the -child.total_value scores below are in.
    """
    children = node.children
    explored = 0.0
    for child in children.values():
        if child.visits:
            explored += child.prior
    parent = node.total_value / node.visits if node.visits else 0.0
    fpu = parent - reduction * math.sqrt(explored)

    best_score = -math.inf
    best = None
    sqrt_visits = math.sqrt(node.visits)
    for item in children.items():
        child = item[1]
        n = child.visits
        score = ((-(child.total_value / n) if n else fpu)
                 + c_puct * child.prior * sqrt_visits / (1 + n))
        if score > best_score:
            best_score = score
            best = item
    return best


def add_virtual_loss(path):
    for node in path[1:]:
        node.visits += VIRTUAL_LOSS
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


# ---------------------------------------------------------------- network


_device_cache = {}


def device_for(model):
    key = id(model)
    if key not in _device_cache:
        device = pick_device()
        model.to(device)
        _device_cache[key] = device
    return _device_cache[key]


@torch.no_grad()
def evaluate_batch(model, boards, device):
    """Policy logits as a (B, 4672) numpy array, and values as a list."""
    x = encode_boards(boards, np.uint8).to(device).float()
    use_amp = device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        logits, values = model(x)
    return (logits.float().cpu().numpy(),
            values.float().cpu().reshape(-1).tolist())


@torch.no_grad()
def evaluate_leaves(model, boards, index_rows, device, smoothing=0.0):
    """
    (priors, values) for a batch of leaves, where index_rows[i] holds the
    policy indices of board i's legal moves.

    Every underpromotion has its own index, so there is nothing to collapse
    together. The legal moves' logits are picked out and softmaxed on the
    device: a few numbers per legal move come back instead of all 4672
    logits per position, and the CPU - the search's bottleneck - does none
    of the work.
    """
    on_device = (PRIORS_ON_DEVICE if PRIORS_ON_DEVICE is not None
                 else device.type == "cuda")
    if not on_device:
        logits, values = evaluate_batch(model, boards, device)
        priors = []
        for row, indices in zip(logits, index_rows):
            p = F.softmax(torch.from_numpy(row[indices]), dim=0)
            if smoothing:
                p = (1.0 - smoothing) * p + smoothing * (1.0 / len(indices))
            priors.append(p.tolist())
        return priors, values

    lengths = [len(row) for row in index_rows]
    width = max(lengths)
    index = torch.tensor([row + [0] * (width - len(row))
                          for row in index_rows])
    count = torch.tensor(lengths)
    x = encode_boards(boards, np.uint8).to(device).float()
    index, count = index.to(device), count.to(device)

    use_amp = device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        logits, values = model(x)

    padding = torch.arange(width, device=device) >= count[:, None]
    picked = logits.float().gather(1, index).masked_fill(padding, -math.inf)
    priors = torch.softmax(picked, dim=1)
    if smoothing:
        priors = (1.0 - smoothing) * priors + smoothing / count[:, None]

    rows = torch.cat([values.float().reshape(-1, 1), priors], 1).cpu().tolist()
    return ([row[1:1 + n] for row, n in zip(rows, lengths)],
            [row[0] for row in rows])


def leaf_value(net_value, board, material):
    if not material:
        return net_value
    balance = math.tanh(material_balance(board) / MATERIAL_SCALE)
    return (1.0 - material) * net_value + material * balance


def make_evaluator(model, device, material, syzygy):
    """The per-level callback the quiescence search uses."""
    @torch.no_grad()
    def evaluate(boards):
        values = [None] * len(boards)
        needed, needed_index = [], []

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
                values[i] = leaf_value(raw[k], needed[k], material)

        return values

    return evaluate


# ---------------------------------------------------------------- cache

# Positions the network has already seen, with their legal moves, priors and
# value. In a game roughly a third of each search's positions were evaluated
# by an earlier search, and a few more are transpositions within it; a hit
# skips move generation and the network. Only facts that depend on nothing
# but the position are kept - repetitions and the fifty-move count are
# always checked afresh, since the same position can be drawn in one line
# and not another.
CACHE_SIZE = 200_000     # entries, roughly 1KB each; 0 turns the cache off

_POSITION = struct.Struct("<9QBb")
_cache = {}
_cache_owner = [None]

# Cached moves are stored packed into 16 bits - from | to << 6 | promotion
# << 12 - and turned back into Move objects through this table, which is
# cheaper both ways than keeping and hashing the objects themselves.
_MOVE_OF = [chess.Move(p & 63, (p >> 6) & 63, (p >> 12) or None)
            for p in range(6 << 12)]


def position_key(board):
    """Everything the network and move generation depend on, and no more."""
    return _POSITION.pack(board.pawns, board.knights, board.bishops,
                          board.rooks, board.queens, board.kings,
                          board.occupied_co[chess.WHITE],
                          board.occupied_co[chess.BLACK],
                          board.castling_rights, board.turn,
                          -1 if board.ep_square is None else board.ep_square)


def claim_cache(model, smoothing, syzygy):
    """Start the cache afresh if anything its entries depend on changed."""
    # Priors depend on how castling is looked up, so that is part of it too.
    owner = (model, smoothing, syzygy, encoding_v2.CASTLE_ONTO_ROOK)
    if _cache_owner[0] is None or any(
            a is not b and a != b for a, b in zip(_cache_owner[0], owner)):
        _cache.clear()
        _cache_owner[0] = owner


def remember(key, moves, priors, value):
    if len(_cache) >= CACHE_SIZE:
        # Dicts keep insertion order, so the front is the oldest.
        for old in list(itertools.islice(_cache, CACHE_SIZE // 10)):
            del _cache[old]
    packed = array("H", [m.from_square | m.to_square << 6
                         | (m.promotion or 0) << 12 for m in moves])
    _cache[key] = (packed, array("f", priors), value)


def clear_cache():
    _cache.clear()


# ---------------------------------------------------------------- search


@torch.no_grad()
def run_search(model, root_board, sims, contempt=DRAW_CONTEMPT,
               material=MATERIAL_WEIGHT, c_puct=C_PUCT,
               batch_size=DEFAULT_BATCH, quiesce=QUIESCE_DEPTH,
               syzygy=SYZYGY_PATH, smoothing=POLICY_SMOOTHING,
               forced_depth=FORCED_VISIT_DEPTH, deadline=None,
               min_sims=MIN_SIMS_BEFORE_DEADLINE, fpu=FPU_REDUCTION):
    """
    Run up to `sims` simulations, stopping early if `deadline` passes.

    A deadline lets the caller spend a time budget rather than guessing how
    many simulations fit in it - the difference between using the whole
    clock and finishing early because the rate estimate was low.
    """
    device = device_for(model)
    root = Node()

    # Quiescence needs the network on positions the cache never sees, so it
    # searches without one rather than mixing the two.
    use_cache = CACHE_SIZE > 0 and not quiesce
    if use_cache:
        claim_cache(model, smoothing, syzygy)

    # Every simulation copies the root, and a full game's move list made that
    # a sixth of the search. Only moves since the last capture or pawn move
    # can ever repeat, so those are all repetition detection needs.
    root_board = root_board.copy(stack=root_board.halfmove_clock)

    root_moves, root_index = policy_indices(
        root_board, list(root_board.generate_legal_moves()))
    root_priors, _ = evaluate_leaves(model, [root_board], [root_index],
                                     device, smoothing)
    root.children = dict(zip(root_moves, map(Node, root_priors[0])))

    done = 0
    while done < sims:
        # A deadline never cuts the search below min_sims: returning an
        # unsearched root would mean playing an essentially random move.
        if (deadline is not None and done >= min_sims
                and time.time() >= deadline):
            break
        want = min(batch_size, sims - done)
        pending = []
        seen = set()
        attempts = 0

        while len(pending) < want and attempts < want * 3:
            attempts += 1
            node = root
            path = [node]
            line = []

            # Choosing a path needs only the visit statistics, so the board
            # is left alone until the leaf is known to be worth evaluating.
            depth = 0
            while node.children:
                picked = None
                if depth < forced_depth:
                    picked = select_unvisited(node)
                if picked is None:
                    picked = (select_child(node, c_puct) if fpu is None
                              else select_child_fpu(node, c_puct, fpu))
                move, node = picked
                line.append(move)
                path.append(node)
                depth += 1

            add_virtual_loss(path)

            # A leaf already waiting in this batch was checked when it was
            # queued - it is neither finished nor in the tablebase - so a
            # collision can skip straight to being discarded.
            if id(node) in seen:
                remove_virtual_loss(path)
                continue

            board = root_board.copy()
            for move in line:
                board.push(move)

            key = position_key(board) if use_cache else id(node)
            hit = _cache.get(key) if use_cache else None
            if hit is not None:
                # Being cached means the position already proved to have
                # moves, no mate, no dead draw and no tablebase answer; only
                # the draws that depend on history can differ this time.
                if CLAIM_DRAWS:
                    value = terminal_value(board, contempt)
                elif board.halfmove_clock >= 100 or board.is_repetition(3):
                    value = -contempt
                else:
                    value = None
            else:
                moves = list(board.generate_legal_moves())
                if CLAIM_DRAWS:
                    value = terminal_value(board, contempt)
                else:
                    value = leaf_terminal(board, moves, contempt)
                if value is None and syzygy:
                    value = probe_tablebase(board, contempt, syzygy)
            if value is not None:
                remove_virtual_loss(path)
                backup(path, value)
                done += 1
                continue

            seen.add(id(node))
            if hit is not None:
                pending.append((node, board, path,
                                [_MOVE_OF[p] for p in hit[0]], None, key, hit))
            else:
                kept, indices = policy_indices(board, moves)
                pending.append((node, board, path, kept, indices, key, None))

        if not pending:
            if attempts >= want * 3:
                break
            continue

        # Positions not already known go to the network once each, however
        # many leaves in this batch reached them by different move orders.
        todo = {}
        for entry in pending:
            if entry[6] is None:
                todo.setdefault(entry[5], entry)
        fresh = {}
        if todo:
            todo = list(todo.values())
            priors, raw = evaluate_leaves(model, [e[1] for e in todo],
                                          [e[4] for e in todo],
                                          device, smoothing)
            for entry, p, v in zip(todo, priors, raw):
                fresh[entry[5]] = (p, v)
                if use_cache:
                    remember(entry[5], entry[3], p, v)

        results = [(entry[6][1], entry[6][2]) if entry[6] is not None
                   else fresh[entry[5]] for entry in pending]
        boards = [entry[1] for entry in pending]
        if material:
            base = [leaf_value(results[i][1], boards[i], material)
                    for i in range(len(pending))]
        else:
            base = [r[1] for r in results]

        if quiesce:
            values = qsearch(make_evaluator(model, device, material, syzygy),
                             boards, quiesce, contempt, stand_pat=base)
        else:
            values = base

        for i, entry in enumerate(pending):
            entry[0].children = dict(zip(entry[3], map(Node, results[i][0])))
            remove_virtual_loss(entry[2])
            backup(entry[2], values[i])
            done += 1

    return root


def visit_count(root):
    """Total simulations actually run, for reporting."""
    return int(sum(child.visits for child in root.children.values()))


def choose(root):
    ranked = sorted(root.children.items(), key=lambda kv: kv[1].visits,
                    reverse=True)
    return ranked[0][0], ranked


# ---------------------------------------------------------------- runner


def arg(name, default, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def load(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if ckpt.get("encoding") != "v2":
        print(f"{ckpt_path} is not a v2 checkpoint - use search_batched.py "
              f"for the older nets")
        sys.exit(1)
    model = ChessNetV2(ckpt.get("blocks", 12), ckpt.get("channels", 256))
    model.load_state_dict(ckpt["model"])
    model.eval()
    fuse_batchnorm(model)
    return model, ckpt


def fuse_batchnorm(model):
    """
    Fold every BatchNorm into the convolution before it.

    In eval mode a BatchNorm is a fixed per-channel scale and shift, which
    the convolution's weights and bias can absorb - the same function with
    27 fewer operations per forward pass. Inference only: the fused model
    must never be trained or saved.
    """
    def fuse_sequential(seq):
        seq[0] = fuse_conv_bn_eval(seq[0], seq[1])
        seq[1] = torch.nn.Identity()

    fuse_sequential(model.stem)
    fuse_sequential(model.policy)
    fuse_sequential(model.value_conv)
    for block in model.blocks:
        block.conv1 = fuse_conv_bn_eval(block.conv1, block.bn1)
        block.bn1 = torch.nn.Identity()
        block.conv2 = fuse_conv_bn_eval(block.conv2, block.bn2)
        block.bn2 = torch.nn.Identity()


def settings_from_args():
    return {
        "sims": arg("--sims", DEFAULT_SIMS, int),
        "batch": arg("--batch", DEFAULT_BATCH, int),
        "contempt": arg("--contempt", DRAW_CONTEMPT, float),
        "material": arg("--material", MATERIAL_WEIGHT, float),
        "cpuct": arg("--cpuct", C_PUCT, float),
        "quiesce": arg("--quiesce", QUIESCE_DEPTH, int),
        "smooth": arg("--smooth", POLICY_SMOOTHING, float),
        "forced": arg("--forced", FORCED_VISIT_DEPTH, int),
        "syzygy": arg("--syzygy", SYZYGY_PATH),
        "fpu": arg("--fpu", FPU_REDUCTION, float),
    }


def think(model, board, s):
    return run_search(model, board, s["sims"], s["contempt"], s["material"],
                      s["cpuct"], batch_size=s["batch"], quiesce=s["quiesce"],
                      syzygy=s["syzygy"], smoothing=s["smooth"],
                      forced_depth=s["forced"], fpu=s["fpu"])


def show(board, human_is_white):
    print()
    print(board if human_is_white
          else board.mirror().transform(chess.flip_vertical))
    print()


def parse_move(board, text):
    for parser in (board.parse_san, board.parse_uci):
        try:
            return parser(text)
        except ValueError:
            continue
    return None


def bench():
    ckpt_path = arg("--ckpt", DEFAULT_CKPT)
    s = settings_from_args()
    model, ckpt = load(ckpt_path)
    device = device_for(model)

    board = chess.Board(
        "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9")

    think(model, board, dict(s, sims=200))          # warm the kernels up
    start = time.time()
    root = think(model, board, s)
    elapsed = time.time() - start

    move, ranked = choose(root)
    print(f"\n{ckpt_path}  step {ckpt.get('step', 0):,}")
    print(f"{s['sims']} sims in {elapsed:.2f}s  "
          f"({s['sims']/elapsed:.0f} sims/s on {device})")
    print("  picks " + board.san(move) + ", top three: " + ", ".join(
        f"{board.san(m)} {int(n.visits)}v {-n.value:+.2f}"
        for m, n in ranked[:3]))


def play():
    ckpt_path = arg("--ckpt", DEFAULT_CKPT)
    s = settings_from_args()
    human_is_white = "--black" not in sys.argv

    if not os.path.exists(ckpt_path):
        print(f"no checkpoint at {ckpt_path}")
        sys.exit(1)

    model, ckpt = load(ckpt_path)
    device = device_for(model)

    print(f"{ckpt_path}  ({ckpt.get('blocks', 12)}x{ckpt.get('channels', 256)}"
          f", step {ckpt.get('step', 0):,})")
    print(f"{s['sims']} sims per move on {device}, batch {s['batch']}")
    print(f"cpuct {s['cpuct']}  |  contempt {s['contempt']}  |  "
          f"material {s['material']}  |  quiesce {s['quiesce']}")
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
            root = think(model, board, s)
            elapsed = time.time() - start
            move, ranked = choose(root)
            san = board.san(move)
            summary = ", ".join(
                f"{board.san(m)} {int(n.visits)}v {-n.value:+.2f}"
                for m, n in ranked[:3])
            board.push(move)
            print(f"\nbot plays {san}   ({elapsed:.1f}s, "
                  f"{s['sims']/elapsed:.0f} sims/s)")
            print(f"  searched: {summary}")

        show(board, human_is_white)

    print("game over:", board.result(claim_draw=True))


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        play()
