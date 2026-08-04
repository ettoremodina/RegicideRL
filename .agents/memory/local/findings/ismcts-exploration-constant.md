---
summary: "C=14.14 leaves root visit counts nearly uniform at 500-3000 iterations, so the visit-count choice is close to arbitrary."
created-at: 2026-08-03T22:18:03.0908793Z
updated-at: 2026-08-03T22:18:03.0908793Z
---

# ISMCTS exploration constant

The ISMCTS agent is configured with `exploration_constant: 14.14` (about 10x the
textbook sqrt(2)) in the `experimental_report` section of `config.yaml`.

Traces recorded with `scripts/trace_ismcts.py` on a mid-game solo position show
what that costs. The rewards backed up by the search live in a narrow band: the
mean values of the 13 root actions differed by roughly 0.05. The exploration
term C * sqrt(ln(a)/v) is about 2.6 at 3000 iterations, over an order of
magnitude larger than that spread.

Consequences measured on the same decision (seed 20260803, 13 legal actions):

- C = 14.14, 3000 iterations: root visits 238/236/236/234/234/232 - essentially
  uniform, so `_best_root_action`, which picks the most visited child, is
  choosing on noise.
- C = 1.414, 3000 iterations: root visits 309 down to 146, and the ordering by
  visits matches the ordering by mean reward.
- C = 1.414, 500 iterations: 45 down to 30, already a usable gradient.

This does not prove 1.414 wins more games; the published win rates were measured
with 14.14. It does mean the reported strength of ISMCTS cannot be attributed to
selective search at these budgets, and that a C sweep is worth running before
any further tuning of iteration counts.
