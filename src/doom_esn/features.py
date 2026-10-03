"""Memory back-ends (reservoir vs delay window) and the whitening projection.

``WindowMemory`` is the classical null from credence (``delay_mlp``): the last ``window``
encoded frames, i.e. an explicit delay embedding. ``Whitener`` projects any memory output onto
its top-k principal axes and scales to unit variance, which is the decorrelated, low-dimension
state that a diagonal search (CMA-ES) and a linear readout can use (docs/MATHS.md, section 5.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ._types import F32
from .reservoir import ReservoirWeights


class Memory(Protocol):
    dim: int

    def reset(self) -> None: ...
    def step(self, u: F32, dt: float) -> F32: ...


class ReservoirMemory:
    def __init__(self, weights: ReservoirWeights) -> None:
        self.weights, self.dim = weights, weights.n
        self._x: F32 = weights.zeros()

    def reset(self) -> None:
        self._x = self.weights.zeros()

    def step(self, u: F32, dt: float) -> F32:
        self._x = self.weights.step(self._x, u, dt)
        return self._x.copy()


class WindowMemory:
    """Last ``window`` inputs, newest first; zeros (= the mean image) before the start."""

    def __init__(self, d_in: int, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.d_in, self.window, self.dim = d_in, window, d_in * window
        self._buf = np.zeros((window, d_in), np.float32)

    def reset(self) -> None:
        self._buf[:] = 0.0

    def step(self, u: F32, dt: float) -> F32:
        self._buf[1:] = self._buf[:-1]
        self._buf[0] = u
        return self._buf.ravel().copy()


@dataclass(frozen=True)
class Whitener:
    mean: F32
    V: F32  # (d, k) principal axes
    scale: F32  # (k,) per-axis std

    @property
    def k(self) -> int:
        return int(self.V.shape[1])

    @classmethod
    def fit(cls, X: F32, k: int) -> Whitener:
        if X.shape[0] < k + 2:
            raise ValueError(f"need more than k+1={k + 1} samples, got {X.shape[0]}")
        mean = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
        k = min(k, len(s))
        sd = s[:k] / np.sqrt(X.shape[0] - 1)
        return cls(
            mean.astype(np.float32),
            vt[:k].T.astype(np.float32),
            np.maximum(sd, 1e-6).astype(np.float32),
        )

    def __call__(self, x: F32) -> F32:
        return (((x - self.mean) @ self.V) / self.scale).astype(np.float32)

    def to_arrays(self) -> dict[str, Any]:
        return {"wh_mean": self.mean, "wh_V": self.V, "wh_scale": self.scale}

    @classmethod
    def from_arrays(cls, arrs: Any) -> Whitener:
        return cls(arrs["wh_mean"], arrs["wh_V"], arrs["wh_scale"])
