"""Health-gated reservoir calibration on random-policy frames (no reward involved).

Because the fitness / RL loss cannot repair a bad reservoir (docs/MATHS.md, 5.1), the
reservoir is chosen *before* learning by data-only criteria:

  * contractivity:  conditional Lyapunov exponent  lambda <= max_lyapunov  (echo state)
  * memory horizon: -1/lambda in [min_horizon, max_horizon] steps (delay window, 2.3)
  * richness:       maximise effective rank of the state covariance
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np

from ._types import F32
from .diagnostics import RunningMoments, StateHealth, conditional_lyapunov
from .reservoir import ReservoirConfig, ReservoirWeights


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CalibrationResult:
    cfg: ReservoirConfig
    health: StateHealth
    lyapunov: float

    @property
    def horizon(self) -> float:
        return float("inf") if self.lyapunov >= 0 else -1.0 / self.lyapunov


def evaluate_config(
    cfg: ReservoirConfig, inputs: F32, dt: float, burn: int = 100
) -> CalibrationResult:
    w = ReservoirWeights.build(cfg, inputs.shape[1])
    x = w.zeros()
    moments = RunningMoments(w.n)
    for t, u in enumerate(inputs):
        x = w.step(x, u, dt)
        if t >= burn:
            moments.update(x)
    lam = conditional_lyapunov(w, inputs, dt=dt, burn=burn)
    return CalibrationResult(cfg, StateHealth.from_moments(moments), lam)


def grid_search(
    base: ReservoirConfig,
    inputs: F32,
    dt: float,
    spectral_radii: Iterable[float],
    input_scalings: Iterable[float],
    taus: Iterable[float],
    burn: int = 100,
) -> list[CalibrationResult]:
    return [
        evaluate_config(
            replace(base, spectral_radius=rho, input_scaling=g, tau=tau), inputs, dt, burn
        )
        for rho, g, tau in itertools.product(spectral_radii, input_scalings, taus)
    ]


def select(
    results: list[CalibrationResult],
    max_lyapunov: float = -0.02,
    min_horizon: float = 2.0,
    max_horizon: float = 60.0,
) -> CalibrationResult:
    ok = [
        r
        for r in results
        if r.lyapunov <= max_lyapunov
        and min_horizon <= r.horizon <= max_horizon
        and np.isfinite(r.health.erank)
    ]
    if not ok:
        raise CalibrationError(
            f"no configuration passed the health gate (lyapunov<={max_lyapunov}, "
            f"horizon in [{min_horizon}, {max_horizon}]) among {len(results)} candidates"
        )
    return max(ok, key=lambda r: r.health.erank)
