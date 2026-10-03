from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from conftest import dense_weights
from doom_esn.diagnostics import (
    RunningMoments,
    condition_number,
    conditional_lyapunov,
    effective_rank,
    participation_ratio,
)
from doom_esn.reservoir import leak_rate, spectral_norm


def test_effective_rank_and_pr_extremes() -> None:
    assert effective_rank(np.ones(10)) == pytest.approx(10.0)
    assert participation_ratio(np.ones(10)) == pytest.approx(10.0)
    rank1 = np.array([5.0, 0, 0, 0])
    assert effective_rank(rank1) == pytest.approx(1.0)
    assert participation_ratio(rank1) == pytest.approx(1.0)
    assert 1.0 <= effective_rank(np.array([4.0, 2.0, 1.0, 0.5])) <= 4.0


def test_condition_number_ignores_numerical_null_space() -> None:
    assert condition_number(np.array([100.0, 1.0, 1e-20])) == pytest.approx(100.0)


def test_running_moments_match_numpy(rng: np.random.Generator) -> None:
    X = rng.standard_normal((500, 7)) * np.arange(1, 8) + 3.0
    m = RunningMoments(7)
    for chunk in np.array_split(X, 9):
        m.update(chunk)
    np.testing.assert_allclose(m.mean, X.mean(0), rtol=1e-10)
    np.testing.assert_allclose(m.cov(), np.cov(X.T), rtol=1e-9)


def test_lyapunov_matches_log_spectral_radius_of_linearisation(rng: np.random.Generator) -> None:
    """At x=0, u=0, b=0 the tangent map is J=(1-a)I + aW, so lambda -> log rho(J)."""
    n, tau, dt = 60, 6.0, 2.0
    W = rng.standard_normal((n, n)) / np.sqrt(n)
    W *= 0.9 / np.max(np.abs(np.linalg.eigvals(W)))
    w = dense_weights(W, tau=tau)
    a = leak_rate(dt, tau)
    J = (1 - a) * np.eye(n) + a * W.astype(np.float32)
    expected = np.log(np.max(np.abs(np.linalg.eigvals(J))))
    lam = conditional_lyapunov(w, np.zeros((6000, 3), np.float32), dt=dt, burn=500)
    assert lam == pytest.approx(expected, abs=0.01)


def test_contractive_reservoir_forgets_initial_condition(rng: np.random.Generator) -> None:
    """||W||_2 < 1 is sufficient for the echo-state property: two starts must merge."""
    n = 80
    W = rng.standard_normal((n, n))
    W *= 0.8 / np.linalg.norm(W, 2)
    w = dense_weights(W)
    assert spectral_norm(sp.csr_matrix(W.astype(np.float32))) < 0.81
    x, y = w.zeros(), (rng.standard_normal(n)).astype(np.float32)
    gap0 = float(np.linalg.norm(x - y))
    for _ in range(200):
        u = rng.standard_normal(3).astype(np.float32)
        x, y = w.step(x, u, 1.0), w.step(y, u, 1.0)
    assert float(np.linalg.norm(x - y)) < 1e-6 * gap0


def test_participation_ratio_never_exceeds_effective_rank(rng: np.random.Generator) -> None:
    """Renyi entropies are non-increasing in order: H_2 <= H_1, so 1 <= PR <= erank <= n."""
    for _ in range(200):
        lam = rng.exponential(size=int(rng.integers(2, 50))) ** rng.uniform(0.5, 4.0)
        assert 1.0 - 1e-9 <= participation_ratio(lam) <= effective_rank(lam) + 1e-9
        assert effective_rank(lam) <= len(lam) + 1e-9
