"""Run the pre-planned comparison (docs/ANALYSIS_PLAN.md). Appends one JSON line per run."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from doom_esn.config import ESConfig
from doom_esn.encoder import FrameEncoder
from doom_esn.envs import collect_pooled_frames, make_doom_env
from doom_esn.es import evaluate, train_es
from doom_esn.features import Whitener, WindowMemory
from doom_esn.policy import LinearPolicy
from doom_esn.reservoir import ReservoirConfig, ReservoirWeights
from doom_esn.wrappers import FeatureObservation, ReservoirObservation

DT, N_CALIB = 4.0, 1000
ARMS = ["reservoir_raw", "reservoir_white", "window1_white", "window2_white"]
RES = ReservoirConfig(
    n_res=400,
    spectral_radius=1.1,
    density=0.05,
    input_scaling=3.0,
    bias_scaling=0.1,
    tau=4.0,
    seed=0,
)
OUT = Path("docs/results/compare")
OUT.mkdir(parents=True, exist_ok=True)


def collect(wr, n, rng, n_act):
    X, (o, _) = [], wr.reset(seed=int(rng.integers(1 << 30)))
    for _ in range(n):
        X.append(o)
        o, _, te, tr, _ = wr.step(int(rng.integers(n_act)))
        if te or tr:
            o, _ = wr.reset()
    return np.asarray(X, np.float32)


def build_arm(arm, env, enc, U, rng, n_act):
    if arm.startswith("reservoir"):
        w = ReservoirWeights.build(RES, enc.dim)
        wr = ReservoirObservation(env, w, enc, DT)
        if arm == "reservoir_white":
            wr.set_whitener(Whitener.fit(collect(wr, N_CALIB, rng, n_act), 24))
        return wr, (0.05 if arm == "reservoir_raw" else 0.5), arm == "reservoir_raw" or True
    W = 1 if arm == "window1_white" else 2
    proj = Whitener.fit(U, 12)
    wr = FeatureObservation(env, enc, WindowMemory(12, W), DT, input_proj=proj)
    wr.set_whitener(Whitener.fit(collect(wr, N_CALIB, rng, n_act), 12 * W))
    return wr, 0.5, False


def main(seeds):
    env = make_doom_env("VizdoomBasic-v1", 4)
    n_act = int(env.action_space.n)
    for seed in seeds:
        for arm in ARMS:
            t0 = time.time()
            rng = np.random.default_rng(1000 + seed)
            enc = FrameEncoder(24, 32)
            pooled = collect_pooled_frames(env, enc, N_CALIB, seed=seed)
            enc.fit(pooled)
            U = enc.transform_pooled(pooled)
            wr, sigma0, health = build_arm(arm, env, enc, U, rng, n_act)
            cfg = ESConfig(
                popsize=12,
                generations=10,
                episodes=5,
                sigma0=sigma0,
                seed=seed,
                max_steps=2000,
                track_health=health,
            )
            res = train_es(wr, cfg, OUT / f"{arm}_s{seed}")
            wr.moments = None
            pol = LinearPolicy(res.theta, wr.obs_dim, n_act)
            r = evaluate(wr, pol, 30, 10**7, 2000)
            row = dict(
                arm=arm,
                seed=seed,
                heldout_mean=float(r.mean()),
                heldout_sem=float(r.std() / np.sqrt(len(r))),
                n_params=LinearPolicy.n_params(wr.obs_dim, n_act),
                best_train=res.best_mean_return,
                frac_positive=float((r > 0).mean()),
                seconds=round(time.time() - t0, 1),
            )
            with open(OUT / "runs.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or range(5))
