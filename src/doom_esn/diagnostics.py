"""State-collapse diagnostics (docs/MATHS.md, sections 4-5).

The RL / fitness signal never reaches the (fixed) reservoir, so a collapsed reservoir is
invisible to the training objective. These functions make it visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._types import F32, F64
from .reservoir import ReservoirWeights, leak_rate


class RunningMoments:
    """Streaming mean / covariance of state vectors (Chan et al. batch merge, float64)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.count = 0
        self.mean: F64 = np.zeros(n)
        self._m2: F64 = np.zeros((n, n))

    def update(self, x: np.ndarray) -> None:
        x2 = np.atleast_2d(np.asarray(x, dtype=np.float64))
        b = x2.shape[0]
        bm = x2.mean(axis=0)
        d = x2 - bm
        bm2 = d.T @ d
        delta = bm - self.mean
        tot = self.count + b
        self._m2 += bm2 + np.outer(delta, delta) * (self.count * b / tot)
        self.mean += delta * (b / tot)
        self.count = tot

    def cov(self) -> F64:
        if self.count < 2:
            raise ValueError("need at least 2 samples")
        return self._m2 / (self.count - 1)


def effective_rank(eigvals: np.ndarray) -> float:
    """Roy-Vetterli effective rank exp(H(p)), p = lambda / sum(lambda). In [1, n]."""
    lam = np.clip(np.asarray(eigvals, dtype=np.float64), 0.0, None)
    s = lam.sum()
    if s <= 0:
        return 1.0
    p = lam[lam > 0] / s
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(eigvals: np.ndarray) -> float:
    """(sum lambda)^2 / sum lambda^2. Equals n for a flat spectrum, 1 for rank one."""
    lam = np.clip(np.asarray(eigvals, dtype=np.float64), 0.0, None)
    s2 = float((lam**2).sum())
    return float(lam.sum() ** 2 / s2) if s2 > 0 else 1.0


def condition_number(eigvals: np.ndarray, rel_floor: float = 1e-12) -> float:
    """lambda_max / lambda_min over eigenvalues above ``rel_floor * lambda_max``."""
    lam = np.sort(np.clip(np.asarray(eigvals, dtype=np.float64), 0.0, None))[::-1]
    if lam[0] <= 0:
        return float("inf")
    kept = lam[lam >= rel_floor * lam[0]]
    return float(kept[0] / kept[-1])


@dataclass(frozen=True)
class StateHealth:
    n_samples: int
    erank: float
    pr: float
    cond: float

    @classmethod
    def from_moments(cls, m: RunningMoments) -> StateHealth:
        lam = np.linalg.eigvalsh(m.cov())
        return cls(m.count, effective_rank(lam), participation_ratio(lam), condition_number(lam))


def conditional_lyapunov(
    w: ReservoirWeights, inputs: F32, dt: float = 1.0, burn: int = 100, seed: int = 0
) -> float:
    """Largest conditional Lyapunov exponent of the input-driven reservoir (per step).

    Tangent recursion  v <- ((1-a) I + a diag(1 - tanh^2(pre)) W) v  along the actual
    trajectory, renormalised each step; the mean log growth is the exponent. A negative
    value means perturbations fade (echo-state behaviour); -1/lambda is the memory horizon
    in steps.
    """
    if len(inputs) <= burn:
        raise ValueError("need more inputs than burn-in steps")
    a = leak_rate(dt, w.cfg.tau)
    W = w.W.astype(np.float64)
    W_in = w.W_in.astype(np.float64)
    b = w.bias.astype(np.float64)
    x = np.zeros(w.n)
    v = np.random.default_rng(seed).standard_normal(w.n)
    v /= np.linalg.norm(v)
    acc, cnt = 0.0, 0
    for t, u in enumerate(inputs):
        phi = np.tanh(W @ x + W_in @ u.astype(np.float64) + b)
        v = (1.0 - a) * v + a * (1.0 - phi**2) * (W @ v)
        nv = float(np.linalg.norm(v))
        v /= nv
        x = (1.0 - a) * x + a * phi
        if t >= burn:
            acc += np.log(nv)
            cnt += 1
    return acc / cnt
