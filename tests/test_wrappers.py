from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from doom_esn.diagnostics import RunningMoments
from doom_esn.encoder import FrameEncoder
from doom_esn.reservoir import ReservoirConfig, ReservoirWeights
from doom_esn.wrappers import ReservoirObservation


class FakeScreenEnv(gym.Env):  # type: ignore[type-arg]
    """Tiny stand-in for ViZDoom so CI does not need the engine."""

    observation_space = gym.spaces.Dict({"screen": gym.spaces.Box(0, 255, (48, 64, 3), np.uint8)})
    action_space = gym.spaces.Discrete(2)

    def __init__(self) -> None:
        self._t = 0

    def _obs(self) -> dict[str, Any]:
        return {"screen": np.full((48, 64, 3), (self._t * 20) % 255, np.uint8)}

    def reset(self, *, seed: int | None = None, options: Any = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._t += 1
        return self._obs(), 1.0, self._t >= 5, False, {}


def _wrapped(moments: RunningMoments | None = None) -> ReservoirObservation:
    enc = FrameEncoder(6, 8)
    w = ReservoirWeights.build(ReservoirConfig(n_res=40, seed=2), enc.dim)
    return ReservoirObservation(FakeScreenEnv(), w, enc, dt=4.0, moments=moments)


def test_observation_is_reservoir_state_and_reset_clears_memory() -> None:
    env = _wrapped()
    o0, _ = env.reset(seed=0)
    for _ in range(3):
        o, *_ = env.step(0)
    assert o.shape == (40,) and o.dtype == np.float32 and env.observation_space.contains(o)
    o0b, _ = env.reset(seed=0)
    np.testing.assert_array_equal(o0, o0b)


def test_shared_weights_do_not_share_state() -> None:
    enc = FrameEncoder(6, 8)
    w = ReservoirWeights.build(ReservoirConfig(n_res=40, seed=2), enc.dim)
    a, b = (
        ReservoirObservation(FakeScreenEnv(), w, enc),
        ReservoirObservation(FakeScreenEnv(), w, enc),
    )
    a.reset(seed=0)
    b.reset(seed=0)
    oa, *_ = a.step(0)
    ob0, *_ = b.step(0)
    np.testing.assert_array_equal(oa, ob0)
    a.step(0)
    ob1, *_ = b.step(0)
    assert not np.array_equal(oa, ob1)


def test_moments_collect_every_step() -> None:
    m = RunningMoments(40)
    env = _wrapped(m)
    env.reset(seed=0)
    env.step(0)
    assert m.count == 2


def test_rejects_non_dict_observation() -> None:
    env = ReservoirObservation(
        gym.make("CartPole-v1"),
        ReservoirWeights.build(ReservoirConfig(n_res=10, density=0.5), 48),
        FrameEncoder(6, 8),
    )
    with pytest.raises(TypeError):
        env.reset(seed=0)


def test_window_memory_is_exact_delay_embedding() -> None:
    from doom_esn.features import WindowMemory

    m = WindowMemory(d_in=3, window=3)
    outs = [m.step(np.full(3, float(i + 1), np.float32), 1.0) for i in range(4)]
    np.testing.assert_array_equal(outs[0], [1, 1, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(outs[3], [4, 4, 4, 3, 3, 3, 2, 2, 2])
    m.reset()
    np.testing.assert_array_equal(m.step(np.ones(3, np.float32), 1.0)[3:], 0)


def test_whitener_gives_identity_covariance(rng: np.random.Generator) -> None:
    from doom_esn.features import Whitener

    X = (rng.standard_normal((2000, 30)) @ rng.standard_normal((30, 30))).astype(np.float32) + 5.0
    z = np.array([Whitener.fit(X, 6)(x) for x in X[:1]])  # call path works on a single vector
    assert z.shape == (1, 6)
    wh = Whitener.fit(X, 6)
    Z = np.array([wh(x) for x in X])
    np.testing.assert_allclose(np.cov(Z.T), np.eye(6), atol=0.05)
    np.testing.assert_allclose(Z.mean(0), 0, atol=0.05)


def test_feature_observation_with_whitener_changes_dim() -> None:
    from doom_esn.features import Whitener, WindowMemory
    from doom_esn.wrappers import FeatureObservation

    enc = FrameEncoder(6, 8)
    env = FeatureObservation(FakeScreenEnv(), enc, WindowMemory(enc.dim, 2), dt=4.0)
    env.reset(seed=0)
    X = np.array([env.step(0)[0] for _ in range(4)])
    env.set_whitener(Whitener.fit(X, 2))
    o, _ = env.reset(seed=0)
    assert o.shape == (2,) and env.observation_space.shape == (2,)


def test_input_projection_is_applied_before_the_window() -> None:
    from doom_esn.features import Whitener, WindowMemory
    from doom_esn.wrappers import FeatureObservation

    enc = FrameEncoder(6, 8)
    rng = np.random.default_rng(0)
    proj = Whitener.fit(rng.standard_normal((200, enc.dim)).astype(np.float32), 5)
    env = FeatureObservation(FakeScreenEnv(), enc, WindowMemory(5, 2), dt=4.0, input_proj=proj)
    o, _ = env.reset(seed=0)
    assert o.shape == (10,)  # 2 lags x 5 projected axes
    np.testing.assert_array_equal(o[5:], 0.0)  # second lag is zero padding at episode start
