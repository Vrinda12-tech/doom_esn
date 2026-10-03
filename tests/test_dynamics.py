from __future__ import annotations

import numpy as np
import pytest

from conftest import dense_weights
from doom_esn.dynamics import classify, find_fixed_point, jacobian


def _one_dim(beta: float, tau: float = 8.0):  # type: ignore[no-untyped-def]
    w = dense_weights(np.array([[beta, 0.0], [0.0, 0.5]]), tau=tau, d_in=1)
    return w


def test_fixed_point_ignores_leak_but_stability_does_not() -> None:
    """x = tanh(Wx+c) does not contain a; the eigenvalues mu = 1 - a + a*lambda do."""
    W = np.array([[-3.0, 0.0], [0.0, 0.2]])
    x_small = find_fixed_point(dense_weights(W, tau=100.0, d_in=1), np.zeros(1), dt=1.0)
    x_large = find_fixed_point(dense_weights(W, tau=1.0, d_in=1), np.zeros(1), dt=20.0)
    assert x_small is not None and x_large is not None
    np.testing.assert_allclose(x_small.x, x_large.x, atol=1e-9)
    assert x_small.kind == "stable"  # a ~ 0.01: mu ~ 1 - 0.01*4
    assert x_large.kind == "saddle"  # a ~ 1: mu = lambda = -3 (period-doubling), other stays


def test_pitchfork_structure_of_x_equals_tanh_beta_x() -> None:
    """beta=2: fixed points 0 (saddle here) and +-x*, x* = tanh(2 x*) ~ 0.9575 (stable)."""
    w = _one_dim(2.0)
    xs = np.array([[1.0, 0.0], [-1.0, 0.0], [0.05, 0.0]])
    fps = [find_fixed_point(w, np.zeros(1), x0=x0, dt=2.0) for x0 in xs]
    assert all(fp is not None for fp in fps)
    ref = float(np.arctanh(0.9575)) / 2.0
    assert fps[0].x[0] == pytest.approx(0.9575, abs=2e-4) and fps[0].kind == "stable"  # type: ignore[union-attr]
    assert fps[1].x[0] == pytest.approx(-0.9575, abs=2e-4) and fps[1].kind == "stable"  # type: ignore[union-attr]
    assert abs(fps[2].x[0]) < 1e-8 or fps[2].kind == "stable"  # type: ignore[union-attr]
    assert ref > 0


def test_origin_is_a_saddle_when_one_direction_expands() -> None:
    w = _one_dim(2.0)
    fp = find_fixed_point(w, np.zeros(1), x0=np.zeros(2), dt=2.0)
    assert fp is not None and fp.kind == "saddle" and fp.n_unstable == 1
    assert fp.radius > 1.0


def test_classify_and_jacobian_shapes() -> None:
    assert classify(np.array([0.5, -0.9j]))[2] == "stable"
    assert classify(np.array([1.2, 0.3]))[2] == "saddle"
    assert classify(np.array([1.2, -1.5]))[2] == "unstable"
    assert jacobian(np.eye(3), np.zeros(3), 0.5).shape == (3, 3)


def test_stability_disc_equals_modulus_test(rng: np.random.Generator) -> None:
    """|1-a+a*lam| < 1  <=>  lam in the disc centred -(1-a)/a with radius 1/a (derived by hand)."""
    from doom_esn.dynamics import in_stability_disc

    lam = (rng.uniform(-3, 3, 5000) + 1j * rng.uniform(-3, 3, 5000)).astype(np.complex128)
    for a in (0.05, 0.22, 0.63, 0.95, 1.0):
        np.testing.assert_array_equal(np.abs(1 - a + a * lam) < 1, in_stability_disc(lam, a))


def test_unit_disc_is_inside_stability_disc_and_touches_at_one() -> None:
    from doom_esn.dynamics import in_stability_disc

    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    for a in (0.1, 0.5, 0.9):
        assert in_stability_disc(0.999 * np.exp(1j * th), a).all()  # rho < 1 is sufficient
        assert not in_stability_disc(np.array([1.001]), a).any()  # tangent at +1


def test_origin_unstable_count_matches_eigenvalue_modulus() -> None:
    from doom_esn.dynamics import n_unstable_at_origin

    w = dense_weights(np.diag([1.3, 0.4, -2.0, 0.9]), tau=4.0, d_in=1)
    # a = 1 - exp(-1/4) ~ 0.221; mu = 1 - a + a*lam: lam=1.3 -> 1.066 (unstable), lam=-2 -> 0.34
    assert n_unstable_at_origin(w, dt=1.0) == 1
