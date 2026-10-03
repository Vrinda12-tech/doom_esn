from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("vizdoom")
pytestmark = pytest.mark.doom


def test_reservoir_over_real_doom_frames() -> None:
    from doom_esn.encoder import FrameEncoder
    from doom_esn.envs import collect_pooled_frames, make_doom_env
    from doom_esn.reservoir import ReservoirConfig, ReservoirWeights
    from doom_esn.wrappers import ReservoirObservation

    env = make_doom_env("VizdoomBasic-v1", frame_skip=4)
    enc = FrameEncoder(24, 32)
    enc.fit(collect_pooled_frames(env, enc, 60, seed=0))
    w = ReservoirWeights.build(ReservoirConfig(n_res=64), enc.dim)
    wrapped = ReservoirObservation(env, w, enc, dt=4.0)
    obs, _ = wrapped.reset(seed=0)
    obs, *_ = wrapped.step(wrapped.action_space.sample())
    assert obs.shape == (64,) and np.isfinite(obs).all()
    env.close()
