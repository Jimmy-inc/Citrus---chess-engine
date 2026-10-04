# Citrus

A chess engine built from scratch: a convolutional neural network with policy
and value heads, driven by batched Monte Carlo tree search, in the style of
AlphaZero and Leela Chess Zero. Unlike those, it learns by supervision from
Stockfish evaluations rather than by self-play, because self-play
reinforcement learning is out of reach on a student budget.

It plays on Lichess as
[**Jimmy_inc3710**](https://lichess.org/@/Jimmy_inc3710), around 2400 rapid.

## How it works

**The network.** 12 residual blocks of 256 channels, 15M parameters. The
board is always shown from the side to move's point of view (flipped and
colour-swapped when Black is to move), as 17 planes: 12 for pieces, 4 for
castling rights, 1 for en passant. The policy head is convolutional, the
AlphaZero scheme: 4,672 outputs, one per origin square and move type. The
value head is a single tanh output.

**The search.** PUCT Monte Carlo tree search, batched 48 leaves at a time
with virtual loss so the GPU sees full batches, with the exploration
constant tuned by self-play (cpuct 2.5). Draws only count once they happen.
It ponders on the opponent's time through the UCI protocol. Network results are cached between moves, BatchNorm is folded into
the convolutions, and positions are encoded a batch at a time: around 2,700
simulations a second on an M4 Pro, 1,600–2,500 on an RTX A4000 server node.

**Training.**
1. **Supervised base**, on 200M positions from the Lichess evaluation
   database, each with a depth-22 Stockfish score and best move. 7.2 epochs
   produced `v2_best`.
2. **Weakness fine-tuning.** The engine finds its own mistakes, and a
   fine-tune targets them, mixed 25/75 with ordinary data so the rest is
   not forgotten. Two ways of finding them:
   - self-play games, with Stockfish flagging evaluation drops and
     disagreements (`selfplay.py`, `label_weak.py`)
   - mining the evaluation database directly: score every position with
     the net, search the ones where its policy is most surprised by
     Stockfish's move, and keep those where Stockfish confirms the engine's
     own move lost ground (`mine_hard.py`)

## Measured strength

Against Stockfish 19 limited to 1,500 nodes a move, 400 simulations a move,
paired random openings:

| net | games | score | Elo vs that Stockfish |
|---|---|---|---|
| `v2_best` (supervised base) | 1,062 | 43.7% | −44 |
| `mix_best` (+ self-play weakness fine-tune) | 1,114 | 52.6% | +18 |

That is +62 Elo (±21) from the weakness fine-tune. A Leela-data fine-tune
and a fine-tune on mistakes mined from the evaluation database both made the
net weaker. Every measurement, including the search tuning, is in
[RESULTS.md](RESULTS.md).

## Files

**The current (v2) engine**

| file | what it does |
|---|---|
| `encoding_v2.py` | board planes and move indexing; `python encoding_v2.py` self-tests |
| `train_v2.py` | the network, and supervised training on evaluation records |
| `search_v2.py` | batched MCTS; `python search_v2.py` to play against it |
| `uci_v2.py`, `run_engine_v2.sh` | UCI engine for lichess-bot and chess GUIs |
| `match_v2.py` | plays Stockfish, reports Elo with a confidence interval |
| `versus.py` | two versions of the engine head to head, for tuning |
| `bench_search.py` | search speed, comparing two versions of the code |

**Weakness data**

| file | what it does |
|---|---|
| `selfplay.py` | the engine against itself, with its evaluations recorded |
| `label_weak.py` | Stockfish flags blunders and misjudged positions in those games |
| `mine_hard.py` | finds mistakes in the evaluation database without self-play |
| `mix_data.py` | blends weakness positions with ordinary training data |

**Data and shared pieces**

| file | what it does |
|---|---|
| `extract_evals.py`, `extract.py`, `extract_engine.py` | build training records |
| `convert_lc0.py`, `fetch_lc0.py`, `train_lc0.py` | the Leela-data experiment |
| `quiescence.py`, `endgame.py` | capture search; Syzygy tablebase probing |
| `build_book.py` | opening book from Stockfish analysis |
| `status.sh` | one-command progress check on the training cluster |

**History.** `train.py`, `play.py`, `search_batched.py`, `match.py`,
`uci.py` and the rest are version one: a 6×128 net with a dense policy
layer, kept because the version-two design grew out of them.
`idle_trainer/` is a macOS menu-bar app that trains the old net while the
laptop is idle.

`CLAUDE.md` is the project notebook: commands, measurements, and everything
that went wrong along the way.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy python-chess zstandard

python encoding_v2.py                              # self-test
python search_v2.py --ckpt checkpoints/v2_best.pt --sims 2000   # play it
python match_v2.py --games 100 --nodes 1500 --sims 400 --quiet  # vs Stockfish
```

Trained networks and data are not in the repository: they are hundreds of
megabytes to tens of gigabytes. Matches need Stockfish on the `PATH`.
