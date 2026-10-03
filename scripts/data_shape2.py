"""Second look: corrected fitness-noise model (candidate x episode ANOVA), the origin saddle and the
stability disc, blank-input dynamics, and the state dimension the data wants (canonical correlations)."""

from __future__ import annotations

import json

import numpy as np

from doom_esn.diagnostics import conditional_lyapunov
from doom_esn.dynamics import find_fixed_point
from doom_esn.encoder import FrameEncoder
from doom_esn.envs import make_doom_env
from doom_esn.policy import LinearPolicy
from doom_esn.reservoir import ReservoirConfig, ReservoirWeights, leak_rate
from doom_esn.wrappers import ReservoirObservation

out: dict = {}
env = make_doom_env("VizdoomBasic-v1", 4)
enc = FrameEncoder(24, 32)
n_act = int(env.action_space.n)
DT = 4.0
rng = np.random.default_rng(0)

# ---------- data: random play, pooled frames, with episode ids
P, EP, ep = [], [], 0
obs, _ = env.reset(seed=0)
for _ in range(6000):
    P.append(enc.pool(obs["screen"]))
    EP.append(ep)
    obs, _, te, tr, _ = env.step(int(rng.integers(n_act)))
    if te or tr:
        obs, _ = env.reset()
        ep += 1
P, EP = np.asarray(P, np.float32), np.asarray(EP)
tr_mask = EP < np.quantile(EP, 0.7)
enc.fit(P[tr_mask])
U = enc.transform_pooled(P)

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
W = w.W.toarray().astype(np.float64)
lam = np.linalg.eigvals(W)

# ---------- B1: stability disc.  |1-a+a*lam| < 1  <=>  |lam + (1-a)/a| < 1/a
rows = []
for tau in (4.0, 8.0, 16.0):
    a = leak_rate(DT, tau)
    mu = np.abs(1 - a + a * lam)
    in_disc = np.abs(lam + (1 - a) / a) < 1 / a
    rows.append(
        {
            "tau": tau,
            "a": round(a, 3),
            "n_unstable_at_origin": int((mu > 1).sum()),
            "disc_test_agrees": bool(np.array_equal(mu > 1, ~in_disc)),
            "frac_unstable": round(float((mu > 1).mean()), 4),
        }
    )
out["origin_saddle_by_tau_rho1.1"] = rows
out["W_eig_max_abs"] = float(np.abs(lam).max())

# ---------- B2: blank input = autonomous reservoir
zero = np.zeros((4000, enc.dim), np.float32)
x = w.zeros() + rng.uniform(-0.5, 0.5, w.n).astype(np.float32)
tail = []
for t in range(4000):
    xn = w.step(x, zero[t], DT)
    if t >= 3900:
        tail.append(float(np.linalg.norm(xn - x)))
    x = xn
out["autonomous_blank_input"] = {
    "final_step_change_mean": float(np.mean(tail)),
    "state_norm": float(np.linalg.norm(x)),
    "conditional_lyapunov": float(conditional_lyapunov(w, zero, dt=DT, burn=500)),
}
fp_try = [
    find_fixed_point(w, np.zeros(enc.dim), x0=rng.uniform(-1, 1, w.n), dt=DT) for _ in range(20)
]
ok = [f for f in fp_try if f is not None]
out["autonomous_blank_input"]["fixed_points_found_of_20_starts"] = len(ok)
if ok:
    out["autonomous_blank_input"]["kinds"] = {
        k: sum(f.kind == k for f in ok) for k in ("stable", "saddle", "unstable")
    }

# ---------- A: fitness noise with the right model (candidates share episode seeds = CRN)
wrapped = ReservoirObservation(env, w, enc, dt=DT)


def ep_return(theta: np.ndarray, seed: int) -> float:
    pol = LinearPolicy(theta, w.n, n_act)
    o, _ = wrapped.reset(seed=seed)
    tot = 0.0
    for _ in range(2000):
        o, r, te, tru, _ = wrapped.step(pol.act(o))
        tot += r
        if te or tru:
            break
    return tot


n_c, n_e = 16, 12
Rm = np.array(
    [
        [
            ep_return(
                (0.05 * rng.standard_normal(LinearPolicy.n_params(w.n, n_act))).astype(np.float32),
                5000 + e,
            )
            for e in range(n_e)
        ]
        for _ in range(n_c)
    ]
)
gm = Rm.mean()
ss_c = n_e * ((Rm.mean(1) - gm) ** 2).sum()
ss_e = n_c * ((Rm.mean(0) - gm) ** 2).sum()
ss_r = ((Rm - Rm.mean(1, keepdims=True) - Rm.mean(0, keepdims=True) + gm) ** 2).sum()
ms_c, ms_e, ms_r = ss_c / (n_c - 1), ss_e / (n_e - 1), ss_r / ((n_c - 1) * (n_e - 1))
var_c = max((ms_c - ms_r) / n_e, 0.0)
rel = lambda k: float(var_c / (var_c + ms_r / k))
out["fitness_noise_crn"] = {
    "sd_episode_effect": float(np.sqrt(max((ms_e - ms_r) / n_c, 0))),
    "sd_candidate_effect": float(np.sqrt(var_c)),
    "sd_residual": float(np.sqrt(ms_r)),
    "reliability_k3": rel(3),
    "reliability_k12": rel(12),
    "F_candidate": float(ms_c / ms_r),
    "frac_episodes_return_gt_0": float((Rm > 0).mean()),
    "return_quantiles": [float(q) for q in np.quantile(Rm, [0.05, 0.5, 0.95])],
}

# ---------- C: what state dimension does the data want?  canonical correlations past <-> future (PCA coords)
K, p, f = 12, 4, 4
V = np.linalg.eigh(np.cov(U[tr_mask].T))[1][:, ::-1][:, :K]
Z = (U - U[tr_mask].mean(0)) @ V


def pf(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Pp, Ff = [], []
    for e in np.unique(EP[mask]):
        z = Z[EP == e]
        for t in range(p - 1, len(z) - f):
            Pp.append(np.concatenate([z[t - k] for k in range(p)]))
            Ff.append(np.concatenate([z[t + 1 + k] for k in range(f)]))
    return np.asarray(Pp), np.asarray(Ff)


def cca(Pp: np.ndarray, Ff: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    Pp, Ff = Pp - Pp.mean(0), Ff - Ff.mean(0)
    n = len(Pp)
    Spp, Sff, Spf = Pp.T @ Pp / n, Ff.T @ Ff / n, Pp.T @ Ff / n
    inv = lambda S: (
        np.linalg.inv(np.linalg.cholesky(S + ridge * np.trace(S) / len(S) * np.eye(len(S)))).T
    )
    M = inv(Spp).T @ Spf @ inv(Sff)
    return np.linalg.svd(M, compute_uv=False)


Ptr, Ftr = pf(tr_mask)
rho = cca(Ptr, Ftr)
# null: shuffle episodes of the future block to estimate the finite-sample noise floor of the top correlations
rng2 = np.random.default_rng(3)
null_top = [float(cca(Ptr, Ftr[rng2.permutation(len(Ftr))])[0]) for _ in range(20)]
info = -0.5 * np.cumsum(np.log(1 - np.minimum(rho, 0.999999) ** 2))
out["state_dim_from_canonical_correlations"] = {
    "n_pairs": len(Ptr),
    "rho_top12": [round(float(r), 3) for r in rho[:12]],
    "null_rho_top_max": round(max(null_top), 3),
    "n_rho_above_null": int((rho > max(null_top)).sum()),
    "cum_predictive_info_nats_at_k": {
        str(k): round(float(info[k - 1]), 2) for k in (1, 2, 3, 4, 6, 8, 12)
    },
}
json.dump(out, open("docs/results/data_shape2.json", "w"), indent=2)
print(json.dumps(out, indent=2))
