"""
build_book.py - build a Polyglot opening book from Stockfish analysis.

Downloaded books contain moves that are popular rather than best. This walks
the opening tree with Stockfish and keeps only moves it rates within a
tolerance of the best one, so every line in the book is objectively sound.

Usage:
    python build_book.py book.bin
    python build_book.py book.bin --plies 12 --depth 22 --tolerance 20
    python build_book.py book.bin --budget 5000 --width 3

Options:
    --plies      how deep to go, in half-moves                  (default 10)
    --depth      Stockfish search depth per position            (default 22)
    --tolerance  centipawns worse than best a move may be       (default 15)
    --width      most moves kept at any one position             (default 3)
    --budget     stop after analysing this many positions      (default 4000)
    --threads    Stockfish threads                               (default 4)
    --hash       Stockfish hash in MB                          (default 1024)

On breadth: your own replies never branch much, but you have to cover
whatever the opponent plays, so the tree grows roughly as width^(plies/2).
Keeping width small and plies moderate is what makes this finish in hours
rather than weeks. Widen the tolerance instead if you want more variety.

On variety: a book holding one move per position makes your bot perfectly
predictable, and any bad line in it gets replayed every single game. Keeping
several near-equal moves, weighted by evaluation, avoids that at almost no
cost in strength.
"""

import os
import struct
import sys
import time

import chess
import chess.engine
import chess.polyglot


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def encode_move(board, move):
    """
    Polyglot move encoding: to-file, to-rank, from-file, from-rank, promotion
    packed into 16 bits.

    The trap here is castling. Polyglot stores it as king-takes-own-rook, so
    e1g1 has to be written as e1h1, and a GUI reading the book back will
    translate it the other way.
    """
    to_square = move.to_square
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            to_square = chess.square(7, rank)      # h-file rook
        else:
            to_square = chess.square(0, rank)      # a-file rook

    promotion = 0
    if move.promotion:
        promotion = {chess.KNIGHT: 1, chess.BISHOP: 2,
                     chess.ROOK: 3, chess.QUEEN: 4}[move.promotion]

    return (chess.square_file(to_square)
            | (chess.square_rank(to_square) << 3)
            | (chess.square_file(move.from_square) << 6)
            | (chess.square_rank(move.from_square) << 9)
            | (promotion << 12))


def write_book(path, entries):
    """entries: list of (zobrist_key, encoded_move, weight)."""
    entries.sort(key=lambda e: (e[0], -e[2]))
    with open(path, "wb") as out:
        for key, move, weight in entries:
            out.write(struct.pack(">QHHI", key, move, min(weight, 65535), 0))


def score_of(info, board):
    """Centipawns from the side to move's perspective, mates capped."""
    score = info["score"].pov(board.turn)
    if score.is_mate():
        mate = score.mate()
        return 30000 - abs(mate) * 100 if mate > 0 else -30000 + abs(mate) * 100
    return score.score()


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        print(__doc__)
        sys.exit(1)

    out_path = sys.argv[1]
    max_plies = arg("--plies", 10, int)
    depth = arg("--depth", 22, int)
    tolerance = arg("--tolerance", 15, int)
    width = arg("--width", 3, int)
    budget = arg("--budget", 4000, int)
    threads = arg("--threads", 4, int)
    hash_mb = arg("--hash", 1024, int)

    try:
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    except FileNotFoundError:
        print("Stockfish not found. Install it with:  brew install stockfish")
        sys.exit(1)
    engine.configure({"Threads": threads, "Hash": hash_mb})

    print(f"building {out_path}")
    print(f"{max_plies} plies deep, depth {depth} per position, "
          f"tolerance {tolerance}cp, width {width}")
    print(f"budget {budget:,} positions\n")

    entries = []
    seen = set()
    analysed = 0
    start = time.time()

    # Breadth-first so that if the budget runs out, the book is uniformly
    # deep rather than one long spike with nothing beside it.
    frontier = [chess.Board()]

    for ply in range(max_plies):
        if not frontier or analysed >= budget:
            break

        next_frontier = []
        print(f"ply {ply + 1}: {len(frontier):,} positions to analyse")

        for board in frontier:
            if analysed >= budget:
                break

            key = chess.polyglot.zobrist_hash(board)
            if key in seen:
                continue
            seen.add(key)

            infos = engine.analyse(board, chess.engine.Limit(depth=depth),
                                   multipv=width)
            analysed += 1

            if analysed % 100 == 0:
                rate = analysed / (time.time() - start)
                print(f"  {analysed:,}/{budget:,} positions "
                      f"({rate:.1f}/s, {len(entries):,} book moves)")

            scored = []
            for info in infos:
                if "pv" not in info or not info["pv"]:
                    continue
                scored.append((score_of(info, board), info["pv"][0]))
            if not scored:
                continue

            best = max(s for s, _ in scored)
            for cp, move in scored:
                if best - cp > tolerance:
                    continue
                # Weight by how close to best it is, so the search picks the
                # top move most often without ever being fully predictable.
                weight = max(1, int(100 - (best - cp)))
                entries.append((key, encode_move(board, move), weight))

                child = board.copy(stack=False)
                child.push(move)
                if not child.is_game_over():
                    next_frontier.append(child)

        frontier = next_frontier

    engine.quit()
    write_book(out_path, entries)

    elapsed = time.time() - start
    size = os.path.getsize(out_path)
    print(f"\ndone in {elapsed/60:.1f} min")
    print(f"  positions analysed : {analysed:,}")
    print(f"  book moves written : {len(entries):,}")
    print(f"  file size          : {size:,} bytes")

    # Read it straight back, as a check that the encoding round-trips.
    board = chess.Board()
    with chess.polyglot.open_reader(out_path) as reader:
        found = list(reader.find_all(board))
    if found:
        moves = ", ".join(f"{board.san(e.move)} ({e.weight})" for e in found)
        print(f"\n  opening moves in the book: {moves}")
    else:
        print("\n  warning: the start position is not in the book")


if __name__ == "__main__":
    main()
