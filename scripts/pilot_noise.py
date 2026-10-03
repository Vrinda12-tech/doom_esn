"""Pilot: does a whitened low-dimension state give the search any signal at initialisation?
Candidate x episode ANOVA under common random numbers, for several initial scales sigma."""

from __future__ import annotations

import json

import numpy as np

from doom_esn.encoder import FrameEncoder
from doom_esn.envs import collect_pooled_frames, make_doom_env
from doom_esn.features import Whitener, WindowMemory
from doom_esn.policy import LinearPolicy
from doom_esn.reservoir import ReservoirConfig, ReservoirWeights
from doom_esn.wrappers import FeatureObservation, ReservoirObservation

DT, K = 4.0, 12
env = make_doom_env("VizdoomBasic-v1", 4)
enc = FrameEncoder(24, 32)
enc.fit(collect_pooled_frames(env, enc, 1500, seed=0))
n_act = int(env.action_space.n)
rng = np.random.default_rng(0)
cfg = ReservoirConfig(
    n_res=400,
    spectral_radius=1.1,
    density=0.05,
    input_scaling=3.0,
    bias_scaling=0.1,
    tau=4.0,
    seed=0,
)
w = ReservoirWeights.build(cfg, enc.dim)


def collect(wrapped, n=1500):
    X, (o, _) = [], wrapped.reset(seed=1)
    for _ in range(n):
        X.append(o)
        o, _, te, tr, _ = wrapped.step(int(rng.integers(n_act)))
        if te or tr:
            o, _ = wrapped.reset()
    return np.asarray(X, np.float32)


def ep_return(wrapped, theta, seed):
    pol = LinearPolicy(theta, wrapped.obs_dim, n_act)
    o, _ = wrapped.reset(seed=seed)
    tot = 0.0
    for _ in range(2000):
        o, r, te, tr, _ = wrapped.step(pol.act(o))
        tot += r
        if te or tr:
            break
    return tot


def anova(R):
    nc, ne = R.shape
    gm = R.mean()
    ms_c = ne * ((R.mean(1) - gm) ** 2).sum() / (nc - 1)
    ms_r = ((R - R.mean(1, keepdims=True) - R.mean(0, keepdims=True) + gm) ** 2).sum() / (
        (nc - 1) * (ne - 1)
    )
    var_c = max((ms_c - ms_r) / ne, 0.0)
    return dict(
        sd_candidate=float(np.sqrt(var_c)),
        sd_resid=float(np.sqrt(ms_r)),
        F=float(ms_c / ms_r),
        rel_k3=float(var_c / (var_c + ms_r / 3)),
        episodes_for_rel_0p5=(float(ms_r / var_c) if var_c > 0 else None),
        mean=float(R.mean()),
        frac_pos=float((R > 0).mean()),
    )


arms = {
    "window2": FeatureObservation(env, enc, WindowMemory(enc.dim, 2), DT),
    "reservoir": ReservoirObservation(env, w, enc, DT),
}
out = {}
for name, wr in arms.items():
    wr.set_whitener(Whitener.fit(collect(wr), K))
    for sigma in (0.1, 0.5, 2.0):
        nc, ne = 12, 10
        R = np.array(
            [
                [
                    ep_return(
                        wr,
                        (sigma * rng.standard_normal(LinearPolicy.n_params(K, n_act))).astype(
                            np.float32
                        ),
                        7000 + e,
                    )
                    for e in range(ne)
                ]
                for _ in range(nc)
            ]
        )
        out[f"{name}|sigma={sigma}"] = anova(R)
        print(
            name,
            sigma,
            {
                k: (round(v, 3) if v is not None else None)
                for k, v in out[f"{name}|sigma={sigma}"].items()
            },
            flush=True,
        )
json.dump(out, open("docs/results/pilot_noise.json", "w"), indent=2)
