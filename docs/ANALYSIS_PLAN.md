# Analysis plan (written BEFORE the comparison was run)

Style follows `credence/PREREGISTRATION.md`; the statistics are credence's own
`rigorous_validation` module, vendored as `src/doom_esn/credence_stats.py`.

**Unit of replication:** one independent training run (seed). Metric: mean return of that run's
final readout over 30 held-out episodes (seeds disjoint from training). n = 5 runs per arm,
distinct seeds; failed runs would be recorded at the worst value, not dropped.

**Arms** (equal budget: population 12, 5 episodes per candidate, 10 generations = 600 training
episodes; diagonal CMA-ES on a linear argmax readout):

| arm | state fed to the readout | readout params | initial sigma |
|-----|--------------------------|---------------:|--------------:|
| `reservoir_raw` | 400-unit reservoir, no projection (the v0.1 design) | 1604 | 0.05 |
| `reservoir_white` | same reservoir, top-24 whitened axes | 100 | 0.5 |
| `window1_white` | current frame, 12 whitened axes (no memory) | 52 | 0.5 |
| `window2_white` | last 2 frames, 12 axes each, whitened jointly (24) | 100 | 0.5 |

**Planned comparisons (exactly three; two-sided exact permutation test, Holm over the three, alpha 0.05):**

1. `reservoir_white` vs `reservoir_raw`: did the data-driven fix help?
2. `window2_white` vs `window1_white`: does memory matter in this scenario at all?
3. `reservoir_white` vs `window2_white`: does the reservoir beat the classical delay null?

With n = 5 vs 5 the smallest attainable two-sided p is 2/252 = 0.0079 (0.0238 after Holm), so
the design *can* reject. Outcomes are REJECT / INCONCLUSIVE / UNDERPOWERED only: there is no
"no effect" outcome. Effect sizes are reported as difference of means with a percentile
bootstrap CI when n >= the module's minimum, otherwise raw points are shown.

**Not planned (exploratory, will be labelled as such):** anything else, including per-seed stories.
