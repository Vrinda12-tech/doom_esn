# doom-esn

A reservoir (echo-state network) agent for [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
that runs on a CPU with 8 GB of RAM, plus diagnostics for state collapse, fixed points and
search noise. Maths (derived by hand, each claim tagged derived / tested / measured /
unverified) is in [`docs/MATHS.md`](docs/MATHS.md).

> **Status: research prototype, no demonstrated win.** On the easy `VizdoomBasic` scenario no
> design here is statistically distinguishable from any other at the budget I could afford.
> What the project does deliver is a measured picture of *why* (below), tested analysis code,
> and a pre-planned, correctly-powered comparison you can rerun with a bigger budget.

## Pipeline

```
ViZDoom frame -> FrameEncoder (24x32 grey, mean-removed)
  -> [optional per-frame PCA/whitening]  -> Memory: Reservoir | DelayWindow
  -> [optional whitening to k axes]      -> LinearPolicy (argmax)   <- diag-CMA-ES (pycma)
```

`calibrate.py` can pick the reservoir by data-only health criteria (conditional Lyapunov
exponent, memory horizon, effective rank). `dynamics.py` finds fixed points and classifies
them (stable / saddle) and gives the stability-disc test. `credence_stats.py` is the exact
permutation / Holm / decide machinery vendored from the author's own
[credence](https://github.com/Vrinda12-tech/credence) repo.

## Quick start

```bash
pip install -e ".[dev,doom]"          # CPU only
make check                            # ruff + mypy + tests (incl. real ViZDoom frames)
doom-esn train --config configs/basic.yaml --out runs/basic
python scripts/data_shape.py ; python scripts/data_shape2.py      # look at the data first
python scripts/compare.py 0 1 2 3 4 && python scripts/analyze_compare.py
```

## What the data say (`docs/results/*.json`, details in MATHS 6-8)

- **Inputs:** 768-d pooled frames have effective rank 17, participation ratio 7; 90% of variance
  is in 25 axes.
- **State wanted:** past-future canonical correlations drop from 0.79 to 0.50 after the 9th:
  a Markov state of roughly 6-9 dimensions; 2 frames of memory add most of the predictability
  (next-frame skill 0.67 -> 0.80), 4 frames saturate it (0.83).
- **Stability:** at `rho = 1.1` the *autonomous origin* is a saddle (5 of 400 eigenvalues
  outside the stability disc). The *driven* reservoir on real frames is not: 115/115 solved
  fixed points stable, 0/145 sampled steps locally expanding, no multistability seen.
- **The real problem was search noise, not saddles:** with the v0.1 design the initial candidates
  differed by about 7 return points against 154 of episode noise (ranking reliability 0.006;
  ~469 episodes per candidate needed). Whitening + a larger initial scale raised that in a pilot
  to ~10-24 episodes (small sample; order of magnitude only).

## The pre-planned comparison (`docs/ANALYSIS_PLAN.md`, written before the runs)

Held-out mean return per independent run (30 episodes each), 5 seeds per arm, equal budget of
600 training episodes. Random play scores about -140.

| arm | per-seed results | mean |
|-----|------------------|-----:|
| reservoir, raw 400-d (v0.1)   | -15, -140, -87, -29, -173 | -89 |
| reservoir, whitened to 24     | -80, -157, -88, -102, -87 | -103 |
| window of 1 frame (no memory) | -139, -134, -122, -141, -175 | -142 |
| window of 2 frames            | -134, -45, -147, -183, -123 | -126 |

| planned comparison | difference of means | exact permutation p | Holm p | decision |
|--------------------|--------------------:|--------------------:|-------:|----------|
| whitened vs raw reservoir | -14 | 0.69 | 1.0 | INCONCLUSIVE |
| 2-frame vs 1-frame window | +16 | 0.65 | 1.0 | INCONCLUSIVE |
| whitened reservoir vs 2-frame window | +24 | 0.40 | 1.0 | INCONCLUSIVE |

The design could have rejected (smallest attainable Holm p is 0.024); it did not. Between-seed
SD is 20-70 return points, larger than every gap. **Nothing here supports "whitening helps",
"memory helps" or "the reservoir beats the delay null".**

Exploratory (not planned, no test run, treat as hypotheses): both reservoir arms sit nominally
above random while both window arms sit near it. A likely confound: the window policy is *linear*
in 12-24 PCA coordinates, whereas the reservoir supplies random nonlinear features, so memory and
nonlinearity are mixed up. A fairer null is a window passed through random tanh features.

## Limitations (read before citing anything)

- One easy scenario, 600 training episodes per run, n = 5 per arm. Basic needs only ~2-4 frames
  of memory (measured), so it cannot show a reservoir advantage even in principle.
- Fixed-point / contraction checks use random-play data and one reservoir draw, not the trained
  policy's own state distribution (effective rank during evaluation was observed to drop to 4-23).
- Calibration picked the edge of its grid; the grid was not widened.
- PPO on `FeatureObservation` should work but is untested. cd_dynamax was read, not run.

## Next steps, in order of value

1. Window + random tanh features as the null (removes the confound above).
2. A scenario that needs memory (e.g. `VizdoomMyWayHome-v1`) before any claim about reservoirs.
3. A larger budget with `multiprocessing` (6 cores) so n and episodes per candidate can grow;
   use the measured `episodes ~ sigma_eps^2 / sigma_alpha^2` rule to size it.
4. Linear-Gaussian filter memory (MATHS 8.4) if windows stop being enough.

## Engineering

src layout, frozen dataclasses, strict YAML config (unknown keys rejected), seeded everywhere
(including ARPACK's start vector), `ruff` + `mypy` + `pytest` (warnings as errors), pre-commit,
CI matrix 3.10-3.12, `.npz` checkpoints with `allow_pickle=False`. Every maths claim has a
named test or a stored measurement (ledger in MATHS 9).
