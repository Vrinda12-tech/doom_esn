from __future__ import annotations

import numpy as np
import pytest

from conftest import dense_weights
from doom_esn.reservoir import (
    ReservoirConfig,
    ReservoirWeights,
    leak_rate,
    spectral_norm,
    spectral_radius,
)


def test_spectral_radius_is_scaled_to_target() -> None:
    w = ReservoirWeights.build(ReservoirConfig(n_res=300, spectral_radius=0.9, seed=3), d_in=10)
    assert spectral_radius(w.W) == pytest.approx(0.9, rel=1e-3)


def test_build_is_deterministic_and_float32() -> None:
    cfg = ReservoirConfig(n_res=100, seed=7)
    a, b = ReservoirWeights.build(cfg, 5), ReservoirWeights.build(cfg, 5)
    assert (a.W != b.W).nnz == 0
    assert np.array_equal(a.W_in, b.W_in)
    assert a.W.dtype == a.W_in.dtype == np.float32


def test_leak_semigroup_property() -> None:
    """exp-integrator: composing gaps dt1 then dt2 equals one gap dt1+dt2 (zero drive)."""
    w = dense_weights(np.zeros((20, 20)), tau=5.0)
    x0 = np.random.default_rng(0).standard_normal(20).astype(np.float32)
    u = np.zeros(3, np.float32)
    two = w.step(w.step(x0, u, 1.3), u, 2.1)
    one = w.step(x0, u, 3.4)
    np.testing.assert_allclose(two, one, rtol=1e-5, atol=1e-6)
    assert leak_rate(3.4, 5.0) == pytest.approx(1 - np.exp(-3.4 / 5.0))


def test_step_equals_exact_ode_solution_under_zero_order_hold() -> None:
    """tau x' = -x + c  with constant c has x(dt) = c + (x0 - c) exp(-dt/tau)."""
    n, tau, dt = 8, 6.0, 2.5
    w = dense_weights(np.zeros((n, n)), tau=tau)
    c = 0.4
    object.__setattr__(w, "bias", np.full(n, np.arctanh(c), np.float32))
    x0 = np.full(n, 0.9, np.float32)
    exact = c + (x0 - c) * np.exp(-dt / tau)
    np.testing.assert_allclose(w.step(x0, np.zeros(3, np.float32), dt), exact, rtol=1e-5)


def test_batch_matches_single(rng: np.random.Generator) -> None:
    w = ReservoirWeights.build(ReservoirConfig(n_res=50, seed=1), d_in=6)
    X = rng.standard_normal((4, 50)).astype(np.float32)
    U = rng.standard_normal((4, 6)).astype(np.float32)
    batch = w.step(X, U, 2.0)
    for i in range(4):
        np.testing.assert_allclose(batch[i], w.step(X[i], U[i], 2.0), rtol=1e-5, atol=1e-6)


def test_spectral_norm_dominates_spectral_radius() -> None:
    w = ReservoirWeights.build(ReservoirConfig(n_res=200, spectral_radius=0.9, seed=5), d_in=4)
    assert spectral_norm(w.W) >= spectral_radius(w.W) - 1e-6


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        ReservoirConfig(density=0.0)
    with pytest.raises(ValueError):
        leak_rate(-1.0, 4.0)
