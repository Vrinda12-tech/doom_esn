"""Numerical checks of the 'loss blindness' claims in docs/MATHS.md section 5."""

from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _ce_grad(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = _softmax(X @ theta.T)
    p[np.arange(len(y)), y] -= 1.0
    return p.T @ X / len(y)


def test_cross_entropy_gradient_formula_matches_finite_differences(
    rng: np.random.Generator,
) -> None:
    X, y = rng.standard_normal((30, 6)), rng.integers(0, 3, 30)
    th = rng.standard_normal((3, 6))

    def loss(t: np.ndarray) -> float:
        p = _softmax(X @ t.T)
        return float(-np.log(p[np.arange(30), y]).mean())

    g, eps = _ce_grad(th, X, y), 1e-6
    for i, j in [(0, 0), (1, 3), (2, 5)]:
        d = np.zeros_like(th)
        d[i, j] = eps
        assert (loss(th + d) - loss(th - d)) / (2 * eps) == np.float64(g[i, j]) or abs(
            (loss(th + d) - loss(th - d)) / (2 * eps) - g[i, j]
        ) < 1e-7


def test_readout_gradient_has_no_component_in_the_state_null_space(
    rng: np.random.Generator,
) -> None:
    """Lemma 5.2: grad_Theta = G^T X, so every row lies in span of visited states."""
    n, r, N = 40, 5, 300
    X = rng.standard_normal((N, r)) @ rng.standard_normal((r, n))  # rank-r state cloud
    y = rng.integers(0, 4, N)
    g = _ce_grad(rng.standard_normal((4, n)), X, y)
    _, _, vt = np.linalg.svd(X)
    null = vt[r:]
    assert np.abs(g @ null.T).max() < 1e-10


def test_null_space_weights_only_shift_a_constant_per_action(rng: np.random.Generator) -> None:
    """Lemma 5.3: moving Theta inside null(Cov) cannot change logit *differences* across
    states, so a collapsed reservoir wastes (n - rank) search dimensions per action."""
    n, r, N = 30, 4, 200
    mu = rng.standard_normal(n)
    X = mu + rng.standard_normal((N, r)) @ rng.standard_normal((r, n))
    _, _, vt = np.linalg.svd(X - X.mean(0))
    null = vt[r:]
    theta = rng.standard_normal((3, n))
    delta = rng.standard_normal((3, n - r)) @ null
    d0 = (X @ theta.T)[0] - (X @ theta.T)[1]
    d1 = (X @ (theta + delta).T)[0] - (X @ (theta + delta).T)[1]
    np.testing.assert_allclose(d0, d1, atol=1e-9)
