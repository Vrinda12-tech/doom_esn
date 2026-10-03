from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from doom_esn.reservoir import ReservoirConfig, ReservoirWeights


def dense_weights(W: np.ndarray, tau: float = 8.0, d_in: int = 3) -> ReservoirWeights:
    """Hand-built reservoir (explicit W, zero input weights/bias) for exact-maths tests."""
    n = W.shape[0]
    cfg = ReservoirConfig(n_res=n, tau=tau)
    return ReservoirWeights(
        cfg=cfg,
        d_in=d_in,
        W=sp.csr_matrix(W.astype(np.float32)),
        W_in=np.zeros((n, d_in), np.float32),
        bias=np.zeros(n, np.float32),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
