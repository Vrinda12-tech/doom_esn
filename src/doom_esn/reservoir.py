"""Sparse leaky echo-state reservoir (float32, memory-light).

Continuous-time model (see docs/MATHS.md, section 2):

    tau * dx/dt = -x + tanh(W x + W_in u + b)

with the nonlinearity held constant over one frame gap ``dt`` (zero-order hold) the exact
update is the exponential integrator

    x <- (1 - a) x + a tanh(W x + W_in u + b),   a = 1 - exp(-dt / tau).

Because ``a`` is computed from ``dt`` the same weights work for any frame-skip, and a
variable frame gap is handled exactly (not by re-tuning a fixed leak rate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, eigs, svds

from ._types import F32

_DENSE_LIMIT = 3000


def leak_rate(dt: float, tau: float) -> float:
    """Exact leak ``a = 1 - exp(-dt/tau)`` for a frame gap ``dt`` (same units as ``tau``)."""
    if dt <= 0 or tau <= 0:
        raise ValueError(f"dt and tau must be positive, got dt={dt}, tau={tau}")
    return float(-np.expm1(-dt / tau))


def spectral_radius(W: sp.csr_matrix, seed: int = 0) -> float:
    """Largest |eigenvalue|. ARPACK for big matrices, dense fallback for small/stubborn ones.

    ARPACK starts from a random vector, so the start ``v0`` is seeded: builds are reproducible.
    """
    n = W.shape[0]
    if n <= 64:
        return float(np.max(np.abs(np.linalg.eigvals(W.toarray().astype(np.float64)))))
    try:
        v0 = np.random.default_rng(seed).standard_normal(n)
        vals = eigs(
            W.astype(np.float64), k=1, which="LM", return_eigenvectors=False, tol=1e-8, v0=v0
        )
        return float(np.abs(vals[0]))
    except ArpackNoConvergence:
        if n > _DENSE_LIMIT:
            raise
        return float(np.max(np.abs(np.linalg.eigvals(W.toarray().astype(np.float64)))))


def spectral_norm(W: sp.csr_matrix, seed: int = 0) -> float:
    """Largest singular value ||W||_2 (the quantity in the *sufficient* echo-state condition)."""
    n = W.shape[0]
    if n <= 64:
        return float(np.linalg.norm(W.toarray().astype(np.float64), 2))
    v0 = np.random.default_rng(seed).standard_normal(n)
    s = svds(W.astype(np.float64), k=1, return_singular_vectors=False, v0=v0)
    return float(s[0])


@dataclass(frozen=True)
class ReservoirConfig:
    n_res: int = 400
    spectral_radius: float = 0.95
    density: float = 0.05
    input_scaling: float = 0.5  # std of the drive per unit rms input
    bias_scaling: float = 0.1
    tau: float = 8.0  # leak time-constant in env tics
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_res < 2:
            raise ValueError("n_res must be >= 2")
        if not 0 < self.density <= 1:
            raise ValueError("density must be in (0, 1]")
        if self.spectral_radius <= 0 or self.tau <= 0 or self.input_scaling < 0:
            raise ValueError("spectral_radius, tau must be > 0 and input_scaling >= 0")


@dataclass(frozen=True, eq=False)
class ReservoirWeights:
    """Immutable weights. Share one instance across many envs; keep state outside."""

    cfg: ReservoirConfig
    d_in: int
    W: sp.csr_matrix
    W_in: F32
    bias: F32

    @classmethod
    def build(cls, cfg: ReservoirConfig, d_in: int) -> ReservoirWeights:
        rng = np.random.default_rng(cfg.seed)
        n = cfg.n_res
        nnz = max(1, round(cfg.density * n * n))
        flat = rng.choice(n * n, size=nnz, replace=False)
        rows, cols = np.divmod(flat, n)
        W = sp.coo_matrix((rng.standard_normal(nnz), (rows, cols)), shape=(n, n)).tocsr()
        rho = spectral_radius(W, cfg.seed)
        if rho < 1e-12:
            raise ValueError("degenerate reservoir (spectral radius ~ 0); raise density or n_res")
        W = sp.csr_matrix(W * (cfg.spectral_radius / rho)).astype(np.float32)
        W_in = (cfg.input_scaling / np.sqrt(d_in) * rng.standard_normal((n, d_in))).astype(
            np.float32
        )
        bias = (cfg.bias_scaling * rng.standard_normal(n)).astype(np.float32)
        return cls(cfg=cfg, d_in=d_in, W=W, W_in=W_in, bias=bias)

    @property
    def n(self) -> int:
        return self.cfg.n_res

    def zeros(self, batch: int | None = None) -> F32:
        shape = (self.n,) if batch is None else (batch, self.n)
        return np.zeros(shape, dtype=np.float32)

    def step(self, x: F32, u: F32, dt: float = 1.0) -> F32:
        """One update. ``x`` is (n,) or (B, n); ``u`` is (d_in,) or (B, d_in)."""
        a = leak_rate(dt, self.cfg.tau)
        x2 = np.atleast_2d(x)
        u2 = np.atleast_2d(u)
        pre = np.asarray(self.W @ x2.T).T + u2 @ self.W_in.T + self.bias
        out = (1.0 - a) * x2 + a * np.tanh(pre)
        return out.astype(np.float32).reshape(x.shape)

    def nbytes(self) -> int:
        return int(
            self.W.data.nbytes
            + self.W.indices.nbytes
            + self.W.indptr.nbytes
            + self.W_in.nbytes
            + self.bias.nbytes
        )
