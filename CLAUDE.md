# chessbot

A chess engine built from scratch: a convolutional network with policy and
value heads, driven by batched MCTS, trained by supervision rather than
self-play. It plays on Lichess as **Jimmy_inc3710**.

I am learning this as I go. Explain reasoning rather than just handing me
commands, and tell me when something I am about to do is a bad idea.

---

## Where things stand

Two networks exist, in two different encodings. **They are not
interchangeable** — the v1 scripts cannot load a v2 checkpoint and vice
versa. Everything current is v2.

| checkpoint | encoding | what it is |
|---|---|---|
| `checkpoints/lc0_best.pt` | v2 | **current best.** v2_best fine-tuned on Leela data |
| `checkpoints/v2_best.pt` | v2 | 12×256 trained on 200M Stockfish evals, 7.2 epochs |
| `checkpoints/evalfull_best.pt` | v1 | the old 6×128 net, still what Lichess runs |

**Strength, measured rather than guessed:**

- The old v1 net scored **+17 Elo (95% CI −8 to +42)** over 462 games
  against Stockfish 18 capped at 325 nodes, 400 sims a side. This is the
  baseline every later comparison should use.
- v2 (step 1.26M, 20k sims) **drew full-strength Stockfish 18 at 8M nodes**
  by threefold repetition, having held +1 to +2.5 through the middlegame.
- lc0_best **drew two games against Stockfish 18 at 4.5M nodes**, 10k sims,
  from the standard opening. Note these were deterministic, so it is one
  game as each colour rather than a sample.

- **mix_best beats v2_best by +62 Elo (±21)**, from ~1,100 games each
  against Stockfish 19 at 1,500 nodes, 400 sims, `--seed 1` (Sep 30):
  mix_best 52.6% (+18 Elo vs SF), v2_best 43.7% (−44). mix_best is v2_best
  fine-tuned 2 epochs on 25% self-play weakness positions + 75% rehearsal.

- **First-play urgency 0.3 is worse: −49 Elo (−77 to −21)** against FPU
  off, 343 games of mix_best against itself with `versus.py`, 800 sims,
  seed 1 (Oct 2). FPU stays off (`search_v2.FPU_REDUCTION = None`).

- **The castling fix did not show a gain against Stockfish:** mix_best
  scored 49.9% with it against long_mix2's 52.6% over the same 1,114
  seed-1 openings (−19 Elo, ±21, so not conclusive either way; Oct 2).
  The two runs also differ in code version (BN folding, cache), so later
  games are not perfectly paired. A direct `versus.py` test is the next
  step before deciding whether the search should keep the fix.

**The known weakness, seen repeatedly: conversion.** The net reaches
winning positions and fails to finish them, usually shuffling into a
repetition. Traced once with `analyse.py` to the value head misjudging
resulting positions, not to the search failing to find moves. Rook and
minor-piece endgames are the worst case.

---

## Two machines

**MacBook Pro M4 Pro, 24GB** — `~/Desktop/chessbot`, venv at `.venv`.
Runs on MPS at roughly 2,700 sims/s with the v2 net since the Sep 29
search optimisations (1,700 before). This is where games,
matches and Lichess play happen.

**teach.cs cluster** — `ssh UTORID@teach.cs.utoronto.ca`, project at
`~/chessbot`, venv at `.venv`. All training happens here.

- `wolf` is the login host: 48 cores, 1TB RAM, **no GPU**. Never train on it.
- `sinfo` shows partition `general` with six **coral** nodes, two RTX A4000
  (16GB) each.
- Home directory is shared across the cluster, so a venv made on wolf works
  on the compute nodes.
- Quota is ~48GB. It fills up fast; `./status.sh` reports it.
- No per-user GPU limit is enforced: five jobs ran at once on Oct 2. Be
  considerate anyway in term time.

Jobs are submitted with `sbatch`, never `srun`, for anything long — `srun`
dies when the SSH connection drops. Always set `--time`; the partition has
no limit, so a forgotten job holds a GPU indefinitely. Check `squeue`
without a user filter before taking several nodes: it is term time and
classes have priority.

```bash
# a GPU training job
sbatch -p general --gres gpu --time=48:00:00 -o ~/chessbot/logs/NAME.out \
  --wrap "cd ~/chessbot && ./.venv/bin/python -u SCRIPT.py ARGS"

# CPU only
sbatch -p general --time=24:00:00 -o ~/chessbot/logs/NAME.out --wrap "..."

squeue -u UTORID
scancel JOBID
./status.sh          # jobs, disk, data, training, all at once
```

---

## The v2 encoding

`encoding_v2.py` defines it, and has a self-test: `python encoding_v2.py`
should print 11/11.

**Canonical orientation.** The board is flipped and colours swapped when
Black is to move, so the network always sees "me at the bottom". This
halves what it has to learn and removes the need for a side-to-move plane.
17 planes: 12 piece, 4 castling, 1 en passant.

**Convolutional policy, AlphaZero scheme.** 4,672 outputs = 64 origin
squares × 73 move types (56 queen-like: 8 directions × 7 distances; 8
knight; 9 underpromotions). Queen promotions reuse the plain move plane.
`MOVE_LOOKUP[from, to, promo]` gives the index.

This replaced a dense 4,096-way layer that held **8.4M of the old net's
10.3M parameters**. The current 12×256 net has 15.0M parameters with 14.2M
in the trunk and 0.6M in the policy head — which matches how Lc0 and
AlphaZero distribute theirs.

---

## Files

**Core v2 — the current pipeline**

- `encoding_v2.py` — canonical planes, move indexing, self-test
- `train_v2.py` — `ChessNetV2`, trains on Stockfish eval records
- `search_v2.py` — batched MCTS. Also `python search_v2.py` to play it
- `match_v2.py` — plays Stockfish, reports Elo with a confidence interval
- `versus.py` — two versions of the bot head to head: same net with different
  search settings (`--a "fpu=0.3" --b ""`), or two nets. For tuning
  search settings; confirm the winner against Stockfish afterwards
- `uci_v2.py` + `run_engine_v2.sh` — UCI wrapper for lichess-bot
- `convert_lc0.py` — Leela chunks → training records. **Has a self-test**
- `train_lc0.py` — trains against Leela's visit distributions
- `fetch_lc0.py` — download, convert, delete loop within a disk budget
- `status.sh` — one-command progress check

**Weakness data** — what produced `mix_best.pt`

- `selfplay.py` — the net against itself, PGN with its evals
- `label_weak.py` — Stockfish flags blunders and misjudged positions
- `mine_hard.py` — weakness data from `evals_d22.bin` without self-play:
  `score` (value error and policy surprise for every record), `mistakes`
  (searches the most surprising positions; where the bot's move differs,
  Stockfish scores both), `select` (thresholds, writes the subset). Most
  disagreements with Stockfish are equally good moves, not mistakes — only
  count a position if Stockfish says the bot's move lost ground
- `mix_data.py` — blends weakness records with rehearsal data (25/75)
- `bench_search.py` — search speed, old code against new

**v1, still used for the deployed Lichess bot**

`train.py`, `train_evals.py`, `finetune.py`, `search.py`,
`search_batched.py`, `play.py`, `match.py`, `print_match.py`, `analyse.py`,
`uci.py`, `run_engine.sh`

**Shared, encoding-agnostic**

- `quiescence.py` — batched negamax with stand-pat
- `endgame.py` — Syzygy tablebase probe (3-4-5 piece, in `syzygy/`)
- `extract_evals.py`, `extract.py`, `extract_engine.py` — data extraction
- `build_book.py` — Polyglot opening book from Stockfish analysis

**Data on the cluster**

- `data/evals_d22.bin` — 200M Stockfish evals at depth 22, 14.4GB, 72 bytes
  a record. Took 8.3 hours to extract; do not delete casually
- `data/lc0/*.bin` — 57.8M Leela positions, 10.3GB, 192 bytes a record.
  **Deleted Oct 1** to free quota, since fine-tuning on it made the net
  weaker. `fetch_lc0.py` can download and convert it again
- `data/scores/` — `mine_hard.py` results for `mix_best`, ~1GB
- `policy_index.py` — Leela's 1,858 move names, fetched from their repo

---

## Training

**On Stockfish evals** (what produced `v2_best.pt`):

```bash
./.venv/bin/python -u train_v2.py --data data/evals_d22.bin \
    --blocks 12 --channels 256 --steps 0 --lr 5e-4
```

Policy loss is ordinary cross-entropy against one move; it bottoms out near
zero. Roughly 11 steps/s on an A4000, ~9.7 hours an epoch over 200M records.

**On Leela distributions** (what produced `lc0_best.pt`):

```bash
./.venv/bin/python -u train_lc0.py --data 'data/lc0/*.bin' \
    --init checkpoints/v2_best.pt --steps 0
```

Quote the glob or the shell eats it. `--init` starts from an existing v2
checkpoint instead of random weights.

**The loss here does not bottom out at zero.** It is cross-entropy against
a soft target, so its floor is the target distribution's own entropy —
measured at **1.555 nats** for this data. Validation policy loss minus
1.555 is the KL divergence, and that gap is the real progress measure:

- started around 0.31
- under 0.20 is a solid improvement
- under 0.15 is probably near what this data supports

Stop when that gap stops shrinking, not at a fixed epoch count.

```bash
grep validation ~/chessbot/logs/lc0train.out | tail -20
```

---

## Testing

```bash
# the comparable measurement — same settings as the +17 Elo baseline
python match_v2.py --games 100 --nodes 325 --sims 400 --quiet \
    --ckpt checkpoints/lc0_best.pt --pgn games.pgn

# play it yourself
python search_v2.py --ckpt checkpoints/lc0_best.pt --sims 2000

# speed check
python search_v2.py --bench --sims 2000

# realistic speed: replays games, searching every second move, and compares
# two versions of the code (each variant runs in its own process)
python bench_search.py --old ~/chessbot --new ~/chessbot_next
```

`--opening N` sets random opening plies, default 6. **`--opening 0` makes
every game identical**, because both engines are deterministic — useful for
one reproducible study game, useless for measurement.

Converting the value head's output to centipawns: `cp = 400 * atanh(v)`.

---

## Lichess

The bot lives at `~/Desktop/lichess-bot`, own venv, `config.yml` points
`name:` at a launcher script in the chessbot folder. **It runs v2**
(`name: "run_engine_v2.sh"`), and `uci_options` sets
a `Checkpoint`, so it plays the chosen net rather than `uci_v2.py`'s own
default (`v2_best.pt`). On the cluster copy (`~/lichess-bot/config.yml`,
the one in use since Sep 15) that is `checkpoints/mix_best.pt`; the
engine logs `info string loaded checkpoint …` at the start of each game. `search_v2.load`
rejects v1 checkpoints outright, which is better than silently misplaying.

```bash
cd ~/Desktop/lichess-bot && source .venv/bin/activate && python lichess-bot.py
```

Time management searches against a **deadline**, not a fixed simulation
count, so the full clock allocation gets used. `info` lines report measured
`nodes`, `nps` and `time`. A 15% safety margin is deliberately left unspent
to cover network latency. `MOVES_TO_GO = 30` in `uci_v2.py` is a flat
assumption and is the crudest part of the time management.

**Rating history.** A snapshot of `lichess.org/@/Jimmy_inc3710` before each
change of net, so each net's games can be told apart afterwards. `?` means
Lichess still treats the rating as provisional. Rapid is the pool that
matters: matchmaking offers 3–15 minutes plus 1–30 seconds, which Lichess
mostly classes as rapid.

| recorded | net that had been playing | rapid | blitz | classical | bullet | games |
|---|---|---|---|---|---|---|
| 2026-09-29 | `v2_best.pt` (cluster copy, step 2,744,000) | 2282 (237) | 2163 (96) | 2418? (12) | 2138? (10) | 365 |
| 2026-10-02 | `mix_best.pt` (step 12,000), search changing underneath — see below | **2437** (360) | 2262 (115) | 2419? (24) | 2138? (10) | 523 |

The last days on v2: Sep 16, rapid +28 =4 −18 (to 2215); Sep 17, rapid
+23 =1 −5 (to 2282, +67), blitz 7 wins from 7, classical +1 −1. Sep 16
probably includes some `lc0_best.pt` games from before the switch back to v2
that afternoon; Sep 17 is v2 alone.

The mix_best stint, Sep 29 – Oct 2, 158 games: rapid +155 over 123 games
(2282 → 2437), blitz +99 over 19 (2163 → 2262). Not one clean experiment —
three things changed during it: the net (mix_best, measured +62 Elo over
v2_best against Stockfish); the faster search from Sep 30 (~700 → 1,600–
2,500 nps on the cluster); and on Oct 2 the castling fix went live
partway through the day (that day: rapid +21 −8, +25). Classical is
meaningless here: a game was stopped midway against a lower-rated
opponent, and 24 games is still provisional-sized. The bot was stopped
Oct 2 to free its GPU; it also times out a lot, which is still to be
investigated and has probably cost rating throughout.

Rows are only comparable if the bot ran on the same host. From Sep 15 it
runs on the cluster (`~/lichess-bot`, submitted with `sbatch --gres gpu`),
where its median speed was **~700 nps** — measured over 1,287 moves of v2 —
against ~1,700 on the Mac. Since the Sep 30 search optimisations it runs at
roughly 1,600–2,500 nps there. The GPU is not the limit; the single-threaded
Python search on one server core probably is. Games before Sep 15 were
played on the Mac with over twice the search, so they are not a fair
baseline for anything after.

---

## Things that have already gone wrong

Learn from these rather than rediscovering them.

**Training diverged to NaN.** float16 overflowed in the forward pass,
which `GradScaler` does not catch — it only guards gradients. The infinity
poisoned a BatchNorm `running_var`, and because BatchNorm uses *batch*
statistics while training and *running* statistics only when evaluating,
the training loss looked merely unimpressive while validation had been
broken for 19,000 steps. Fixed with **bfloat16** (same exponent range as
fp32), **gradient clipping at 1.0**, and a non-finite loss guard. The
lesson: if a validation number goes strange, stop immediately even when
training loss looks survivable.

**The Leela decoder was wrong three times, and my own tests passed every
time** because they encoded the same assumption as the code. Real data
found all three: a file mirroring that produced king-queen-swapped boards,
a promotion-spelling comparison, and castling written as king-takes-rook
(`e1h1`). Hence `convert_lc0.py --self-test`, which checks that every move
Leela gave visits to is legal in the decoded position. **Run it on any new
archive before converting.** It should report 100%.

**Virtualenvs break when Homebrew upgrades Python.** The symptom is
`init_fs_encoding: failed to get the Python codec`. Fix is
`brew reinstall python@3.12` then rebuild the venv. Use the same
interpreter for both venvs so they don't drift.

**Em-dashes.** Pasting commands sometimes turns `--sims` into `—sims`,
which is silently ignored and falls back to the default. The scripts echo
their parsed settings in the header — read it.

**The search counted draws that never happened.** It scored a position as
drawn as soon as a draw *could be claimed*, but chess.com does not claim
for you, so against a 3200 bot it steered into "draws" while the opponent
simply played on. The search now counts a draw only once it actually
occurs (threefold, fifty moves, insufficient material). The old rule is
still available as `search_v2.CLAIM_DRAWS = True` / `match_v2.py
--claim-draws`, but should not be used for play.

**The nets learned castling on an output the search never read.**
Lichess's database writes castling king-onto-rook (`e1h1`), and
`extract_evals.py` copied that text without checking it against the board,
so every net trained on `evals_d22.bin` learned castling at policy output
308 ("slide three squares east from e1"). python-chess hands the search
`e1g1`, output 307. Measured on mix4_best: 53% of the policy on castling in
a Ruy Lopez where the search saw 0.05% and played something else. Found
Oct 1. Fixed by looking castling up king-onto-rook in `policy_indices`
(`encoding_v2.CASTLE_ONTO_ROOK`, on by default; `match_v2.py
--old-castling` plays the old lookup for comparison) and by
converting python-chess's `e1g1` records to the same spelling in
`train_v2.expand`. The lesson, as with the Leela decoder: text from outside
is a claim, not a move — check it against the board.
**But it had not stopped the bot castling.** Over the same 1,114 match
games the old lookup castled 638 times (Stockfish 640), the fix 585. The
first batch at the root visits every legal move once whatever its prior
(the root has no visits yet, so the exploration term is zero for all of
them, and virtual loss pushes each pick onto the next move), and the value
head then rates castling on its merits. The broken prior only mattered
deeper in the tree. Worth remembering before predicting that a prior bug
will show up in play.

**`srun` dies with the SSH session.** Use `sbatch` for anything long.

---

## Worth doing next

Roughly in order of expected value.

**Measure lc0_best against v2_best.** 100 games at 325 nodes, 400 sims,
same as the baseline. Nothing else should be decided before this, because
validation metrics have misled on this project before.

**Squeeze-and-excitation blocks.** Lc0 added these and measured a real
gain. Cheap in parameters, no disruption to the data pipeline. Best
effort-to-benefit ratio available.

**A WDL value head** — separate win, draw and loss outputs instead of one
tanh scalar. A position that is +1.5 but drawish differs from +1.5 and
winning, and a scalar cannot say so. This targets the conversion weakness
directly.

**History planes.** Lc0 feeds 8 positions, 112 planes. This net sees one.
Every Leela record already contains the history and `convert_lc0.py`
throws seven-eighths of it away. Biggest structural gap, biggest change.

**Targeted endgame data.** Generate rook and minor-piece endgame positions,
label with deep Stockfish, fine-tune. Directly aimed at the known weakness.

**Self-play as a weakness detector, not a teacher.** Play the net against
itself, log positions with large evaluation swings or big Stockfish
disagreements, label those with Stockfish, fine-tune. Full self-play RL is
not viable — Python MCTS caps at ~135 games/hour/GPU against AlphaZero's
44 million games — but using it to *find* bad positions is cheap and aims
at the right target.

**More Leela data.** `fetch_lc0.py` only walks the top-level archives.
The `test80/`, `test90/`, `test91/` directories hold far more, and the
download-convert-delete loop scales to them without needing more disk.
