"""Fixed points, stability and saddles of the input-driven reservoir map.

For a constant input u the update  x <- (1-a) x + a tanh(W x + W_in u + b)  has fixed points
satisfying  x = tanh(W x + W_in u + b)  -- the leak ``a`` cancels, so *where* the fixed points
are does not depend on dt/tau. *Stability* does: the Jacobian is  J = (1-a) I + a D W  with
D = diag(1 - x*^2), whose eigenvalues are  mu = 1 - a + a*lambda,  lambda in eig(D W).
(docs/MATHS.md, section 8.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import root

from ._types import F64
from .reservoir import ReservoirWeights, leak_rate

Kind = Literal["stable", "saddle", "unstable"]


@dataclass(frozen=True)
class FixedPoint:
    x: F64
    radius: float  # max |mu|; < 1 means locally attracting
    n_unstable: int  # number of eigenvalues with |mu| > 1
    kind: Kind


def _dense(w: ReservoirWeights) -> tuple[F64, F64, F64]:
    return (
        w.W.toarray().astype(np.float64),
        w.W_in.astype(np.float64),
        w.bias.astype(np.float64),
    )


def jacobian(W: F64, x: F64, a: float) -> F64:
    """Local Jacobian of the update map at state x (dense)."""
    n = len(x)
    return (1.0 - a) * np.eye(n) + a * (1.0 - x**2)[:, None] * W


def classify(mu: np.ndarray, tol: float = 1e-9) -> tuple[float, int, Kind]:
    mod = np.abs(mu)
    n_unst = int((mod > 1.0 + tol).sum())
    kind: Kind = "stable" if n_unst == 0 else ("unstable" if n_unst == len(mu) else "saddle")
    return float(mod.max()), n_unst, kind


def in_stability_disc(lam: np.ndarray, a: float) -> np.ndarray:
    """|1 - a + a*lam| < 1  <=>  |lam + (1-a)/a| < 1/a : a disc of radius 1/a tangent to the unit
    circle at +1 (a -> 1: the unit disc; a -> 0: the half-plane Re(lam) < 1)."""
    return np.abs(np.asarray(lam) + (1.0 - a) / a) < 1.0 / a


def n_unstable_at_origin(w: ReservoirWeights, dt: float) -> int:
    """Unstable directions of the autonomous map at x=0 (b=0): eigenvalues of W outside the disc."""
    lam = np.linalg.eigvals(w.W.toarray().astype(np.float64))
    return int((~in_stability_disc(lam, leak_rate(dt, w.cfg.tau))).sum())


def find_fixed_point(
    w: ReservoirWeights,
    u: np.ndarray,
    x0: np.ndarray | None = None,
    dt: float = 1.0,
    tol: float = 1e-10,
    restarts: int = 5,
    seed: int = 0,
) -> FixedPoint | None:
    """Newton/hybrid solve of x = tanh(Wx + W_in u + b); ``None`` if no start converged.

    ``None`` means *not found*, not *does not exist*: the first start is ``x0`` (or 0), the rest
    are random in [-1, 1]^n. (A single start at 0 failed on the autonomous reservoir in the
    data study although every random start converged.)
    """
    W, W_in, b = _dense(w)
    c = W_in @ np.asarray(u, dtype=np.float64) + b
    n = w.n

    def resid(x: F64) -> F64:
        return x - np.tanh(W @ x + c)

    def dresid(x: F64) -> F64:
        return np.eye(n) - (1.0 - np.tanh(W @ x + c) ** 2)[:, None] * W

    rng = np.random.default_rng(seed)
    starts = [np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64)]
    starts += [rng.uniform(-1, 1, n) for _ in range(restarts)]
    sol = None
    for s0 in starts:
        cand = root(resid, s0, jac=dresid, method="hybr", tol=tol)
        if cand.success and np.abs(resid(cand.x)).max() <= 1e-7:
            sol = cand
            break
    if sol is None:
        return None
    a = leak_rate(dt, w.cfg.tau)
    mu = np.linalg.eigvals(jacobian(W, sol.x, a))
    radius, n_unst, kind = classify(mu)
    return FixedPoint(sol.x, radius, n_unst, kind)


def local_radius_along(
    w: ReservoirWeights, inputs: np.ndarray, dt: float, every: int = 10, burn: int = 50
) -> np.ndarray:
    """max |eig J_t| at sampled steps of an actual driven trajectory (>1: locally expanding)."""
    W, W_in, b = _dense(w)
    a = leak_rate(dt, w.cfg.tau)
    x = np.zeros(w.n)
    out = []
    for t, u in enumerate(inputs):
        pre = W @ x + W_in @ u.astype(np.float64) + b
        phi = np.tanh(pre)
        if t >= burn and (t - burn) % every == 0:
            J = (1.0 - a) * np.eye(w.n) + a * (1.0 - phi**2)[:, None] * W
            out.append(float(np.abs(np.linalg.eigvals(J)).max()))
        x = (1.0 - a) * x + a * phi
    return np.asarray(out)
