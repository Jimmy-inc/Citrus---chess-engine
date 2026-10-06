# Results

Every measurement behind a decision in this project, with the settings it
was made under. Elo differences come with 95% confidence intervals; an
interval that includes zero means the test did not separate the two sides.

**How the matches work.** Random 6-ply openings, each played twice with
colours swapped, so neither side gets the better openings. `--seed` fixes
the openings, so runs with the same seed meet the same positions. Both
engines are deterministic. Elo is computed from the score as
−400·log10(1/p − 1), with the interval from the per-game variance.

## Strength against Stockfish

Stockfish 19 limited to 1,500 nodes a move, the bot at 400 simulations a
move, seed 1 (`match_v2.py`).

| net / setting | games | score | Elo vs that Stockfish | date |
|---|---|---|---|---|
| `v2_best` — supervised base, 200M positions | 1,062 | 43.7% | −44 (−60 to −28) | Sep 30 |
| `mix_best` — + self-play weakness fine-tune | 1,114 | 52.6% | +18 (+3 to +32) | Sep 30 |
| `mix_best` with the castling fix | 1,200 | 49.9% | −1 (−15 to +14) | Oct 2 |
| `hardprev` — + mined-mistakes fine-tune (preview) | 169 | 39.9% | −71 (−115 to −29) | Oct 4 |

`mix_best` is **+62 Elo (±21) over `v2_best`**. The castling and `hardprev`
rows ran on newer search code with the old draw rule (`--claim-draws`) for
comparability; the castling row is directly comparable with the `mix_best`
row over the same 1,114 openings (−19 ± 21).

## Fine-tuning experiments

| experiment | result |
|---|---|
| Leela (Lc0) data, fine-tuned from `v2_best` | weaker in testing; data deleted |
| Self-play weakness data: 25% weakness + 75% rehearsal, 2 epochs → `mix_best` | **+62 Elo** over `v2_best` |
| Same recipe at 1.5, 2 and 2.5 epochs | −1, −14, −8 Elo vs Stockfish 1,500 nodes (500–635 games each): more epochs gave nothing |
| Mined mistakes, 66k at 10% with rehearsal, 4,000 steps → `hardprev` | **−41 Elo** (−77 to −6) head to head against `mix_best`, 204 games |

The one fine-tune that helped trained on positions from the bot's own
games. Both that hurt trained on positions it rarely reaches: Leela's games,
and the 0.1% of database positions that most surprised its policy.

## Search settings

Head to head with `versus.py`: `mix_best` against itself, one setting
changed.

| setting | sims | games | score | Elo | interval |
|---|---|---|---|---|---|
| first-play urgency 0.3 vs off | 800 | 343 | 43.0% | −49 | −77 to −21 |
| castling lookup: fixed vs old | 400 | 642 | 49.5% | −3 | −21 to +15 |
| cpuct 1.0 vs 2.0 | 800 | 192 | 35.7% | −102 | −146 to −62 |
| cpuct 1.5 vs 2.0 | 800 | 186 | 43.5% | −45 | −84 to −8 |
| **cpuct 2.5 vs 2.0** (seeds 1 + 2) | 800 | 1,082 | 52.1% | **+14** | **+1 to +28** |
| cpuct 3.0 vs 2.0 (seeds 1 + 2) | 800 | 1,064 | 51.3% | +9 | −5 to +23 |
| cpuct 4.0 vs 2.0 | 800 | 559 | 49.3% | −5 | −24 to +14 |
| cpuct 2.5 vs 2.0 (seeds 1 + 2) | 1,600 | 400 | 50.5% | +3 | −21 to +28 |
| cpuct 3.0 vs 2.0 (seeds 1 + 2) | 1,600 | 478 | 49.2% | −6 | −27 to +16 |

Shipped: FPU off, castling fix on (neutral, and the lookup that matches
training), cpuct 2.5. The cpuct curve is flat between 2.0 and 3.0 and
matters less with more search.

## Mining the evaluation database

`mine_hard.py` over `evals_d22.bin` with `mix_best`:

- 199.8M positions scored. Value error: median 0.063, 90th percentile
  0.318, 99th 0.800 (value units; 0.25 ≈ 100cp in a level position).
- The 200k positions whose Stockfish move most surprised the policy were
  searched (800 simulations) and differing moves judged by Stockfish at
  depth 16. Of 153,072 searched: same move 1.8%, a different but equally
  good move (within 10cp) 47.2%, 10–20cp worse 7.8%, **over 20cp worse
  43.2%** (over 150cp: 14.3%).
- The mistakes were mostly opening and middlegame positions: 66% had 21+
  pieces on the board (52% of all positions), 1% had 6 or fewer (6%).

## Search speed

| | before | after |
|---|---|---|
| M4 Pro (MPS), sims/s | ~1,700 | ~2,700 |
| RTX A4000 server node, Lichess play, nps | ~700 | 1,600–2,500 |

From batch board encoding, BatchNorm folded into the convolutions, a
network-evaluation cache shared across moves, and CPU-side clean-up. In a
replay of real games the optimised search chose the same move in 90 of 90
positions as the original.

## Lichess

[Jimmy_inc3710](https://lichess.org/@/Jimmy_inc3710), snapshots before
each change of net:

| date | net that had been playing | rapid | blitz | games |
|---|---|---|---|---|
| Sep 29 | `v2_best` | 2282 | 2163 | 365 |
| Oct 2 | `mix_best` (faster search part-way through) | 2437 | 2262 | 523 |
| Oct 6 | **v2 final**: `mix_best`, cpuct 2.5, castling fix, pondering | **2514** | 2281 | 604 |

Rapid passed 2500 on the final v2 settings: +77 over 78 games after the
Oct 2 snapshot, rating deviation 45.
