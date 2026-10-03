"""Gymnasium wrappers: pixels -> memory features. Stateful per env, weights shared."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from ._types import F32
from .diagnostics import RunningMoments
from .encoder import FrameEncoder
from .features import Memory, ReservoirMemory, Whitener
from .reservoir import ReservoirWeights


class FeatureObservation(gym.Wrapper):
    """Observation := optional-whitener(memory(encoder(frame))).

    ``memory`` may be a reservoir (fading memory) or a delay window (explicit lags): the same
    downstream learner sees either. ``dt`` is the frame gap in tics (``info["dt"]`` overrides).
    ``moments`` (optional) accumulates the *raw* memory output for collapse diagnostics.
    ``input_proj`` (optional) projects each encoded frame to a few whitened axes *before* the
    memory: a delay window must be built from per-frame coordinates, because PCA of stacked
    frames discards the (low-variance) frame-to-frame difference that carries velocity.
    """

    def __init__(
        self,
        env: gym.Env,
        encoder: FrameEncoder,
        memory: Memory,
        dt: float = 1.0,
        whitener: Whitener | None = None,
        moments: RunningMoments | None = None,
        input_proj: Whitener | None = None,
    ) -> None:
        super().__init__(env)
        self.encoder, self.memory, self.dt = encoder, memory, dt
        self.input_proj = input_proj
        self.whitener, self.moments = whitener, moments
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)

    @property
    def memory_dim(self) -> int:
        return int(self.memory.dim)

    @property
    def obs_dim(self) -> int:
        return self.whitener.k if self.whitener is not None else self.memory_dim

    def set_whitener(self, whitener: Whitener | None) -> None:
        self.whitener = whitener
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)

    def _screen(self, obs: Any) -> np.ndarray:
        try:
            return np.asarray(obs["screen"])
        except (KeyError, TypeError, IndexError) as e:
            raise TypeError("wrapped env must return a dict observation with a 'screen' key") from e

    def _advance(self, obs: Any, dt: float) -> F32:
        u = self.encoder(self._screen(obs))
        if self.input_proj is not None:
            u = self.input_proj(u)
        f = self.memory.step(u, dt)
        if self.moments is not None:
            self.moments.update(f)
        return self.whitener(f) if self.whitener is not None else f

    def reset(self, **kwargs: Any) -> tuple[F32, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.memory.reset()
        return self._advance(obs, self.dt), info

    def step(self, action: Any) -> tuple[F32, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        dt = float(info.get("dt", self.dt))
        return self._advance(obs, dt), float(reward), terminated, truncated, info


class ReservoirObservation(FeatureObservation):
    """FeatureObservation with a reservoir memory (kept for the original constructor)."""

    def __init__(
        self,
        env: gym.Env,
        weights: ReservoirWeights,
        encoder: FrameEncoder,
        dt: float = 1.0,
        moments: RunningMoments | None = None,
        whitener: Whitener | None = None,
    ) -> None:
        if encoder.dim != weights.d_in:
            raise ValueError(f"encoder dim {encoder.dim} != reservoir d_in {weights.d_in}")
        super().__init__(env, encoder, ReservoirMemory(weights), dt, whitener, moments)
        self.weights = weights
