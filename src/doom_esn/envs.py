"""ViZDoom environment construction and random-policy data collection."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from ._types import F32
from .encoder import FrameEncoder


def make_doom_env(env_id: str = "VizdoomBasic-v1", frame_skip: int = 4, **kwargs: Any) -> Any:
    try:
        import vizdoom.gymnasium_wrapper  # noqa: F401  (registers the Vizdoom* ids)
    except ImportError as e:  # pragma: no cover
        raise ImportError("ViZDoom missing: pip install 'doom-esn[doom]'") from e
    return gym.make(env_id, frame_skip=frame_skip, **kwargs)


def collect_pooled_frames(env: Any, encoder: FrameEncoder, n_steps: int, seed: int = 0) -> F32:
    """Roll a uniform-random policy and keep only the *pooled* frames (n_steps, dim)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_steps, encoder.dim), dtype=np.float32)
    obs, _ = env.reset(seed=seed)
    for t in range(n_steps):
        out[t] = encoder.pool(obs["screen"])
        obs, _, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
        if term or trunc:
            obs, _ = env.reset()
    return out
