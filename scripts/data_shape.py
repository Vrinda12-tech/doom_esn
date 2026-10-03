"""Look at the data before deciding anything: spectrum, memory demand, fixed points, fitness noise."""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from doom_esn.diagnostics import effective_rank, participation_ratio
from doom_esn.dynamics import find_fixed_point, local_radius_along
from doom_esn.encoder import FrameEncoder
from doom_esn.envs import make_doom_env
from doom_esn.policy import LinearPolicy
from doom_esn.reservoir import ReservoirConfig, ReservoirWeights
from doom_esn.wrappers import ReservoirObservation

SEED, T_STEPS, DT = 0, 6000, 4.0
out: dict = {}
env = make_doom_env("VizdoomBasic-v1", 4)
enc = FrameEncoder(24, 32)
n_act = int(env.action_space.n)

# ---- collect random-play data: pooled frame, action taken, episode id
rng = np.random.default_rng(SEED)
P, A, EP = [], [], []
obs, _ = env.reset(seed=SEED)
ep = 0
for _ in range(T_STEPS):
    a = int(rng.integers(n_act))
    P.append(enc.pool(obs["screen"]))
    A.append(a)
    EP.append(ep)
    obs, _, term, trunc, _ = env.step(a)
    if term or trunc:
        obs, _ = env.reset()
        ep += 1
P, A, EP = np.asarray(P, np.float32), np.asarray(A), np.asarray(EP)
out["n_steps"], out["n_episodes"], out["mean_episode_len"] = T_STEPS, ep + 1, T_STEPS / (ep + 1)

# ---- A1: shape of the raw input
tr = EP < np.quantile(EP, 0.7)
enc.fit(P[tr])
U = enc.transform_pooled(P)
lam = np.linalg.eigvalsh(np.cov(U[tr].T))[::-1]
cum = np.cumsum(lam) / lam.sum()
out["input_shape"] = {
    "dim": int(U.shape[1]),
    "erank": effective_rank(lam),
    "pr": participation_ratio(lam),
    "k90": int(np.searchsorted(cum, 0.90) + 1),
    "k99": int(np.searchsorted(cum, 0.99) + 1),
    "top5_var_frac": [float(v) for v in (lam[:5] / lam.sum())],
}

# ---- A2: how much memory does next-frame prediction want?  (credence's eps(W) on real frames)
K = 12
V = np.linalg.eigh(np.cov(U[tr].T))[1][:, ::-1][:, :K]
Z = (U - U[tr].mean(0)) @ V  # PCA coords, T x K
Aoh = np.eye(n_act, dtype=np.float32)[A]


def ridge_skill(
    X: np.ndarray, Y: np.ndarray, valid: np.ndarray, lam_grid=(1e-2, 1e-1, 1.0, 10.0, 100.0)
):
    """Fit on train episodes, pick lambda on the last 20% of train, report test skill 1-MSE/Var."""
    idx_tr = np.where(valid & tr)[0]
    split = idx_tr[int(0.8 * len(idx_tr))]
    fit, val = idx_tr[idx_tr < split], idx_tr[idx_tr >= split]
    te = np.where(valid & ~tr)[0]
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
    Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    best = None
    for l in lam_grid:
        B = np.linalg.solve(Xs[fit].T @ Xs[fit] + l * np.eye(Xs.shape[1]), Xs[fit].T @ Y[fit])
        e = ((Xs[val] @ B - Y[val]) ** 2).sum() / ((Y[val] - Y[fit].mean(0)) ** 2).sum()
        if best is None or e < best[0]:
            best = (e, l, B)
    _, l, B = best
    B = np.linalg.solve(
        Xs[idx_tr].T @ Xs[idx_tr] + l * np.eye(Xs.shape[1]), Xs[idx_tr].T @ Y[idx_tr]
    )
    return float(1 - ((Xs[te] @ B - Y[te]) ** 2).sum() / ((Y[te] - Y[idx_tr].mean(0)) ** 2).sum())


def lagged(Wn: int):
    cols, valid = [], np.ones(len(Z), bool)
    for k in range(Wn):
        zk = np.roll(Z, k, axis=0)
        ak = np.roll(Aoh, k, axis=0)
        same = np.roll(EP, k) == EP
        valid &= same if k else True
        cols += [zk, ak]
    valid &= np.roll(EP, -1) == EP
    valid[-1] = False
    return np.hstack(cols), valid


Ynext = np.roll(Z, -1, axis=0)
skill = {}
for Wn in (1, 2, 4, 8, 16):
    X, valid = lagged(Wn)
    skill[f"W={Wn}"] = ridge_skill(X, Ynext, valid)
persist_te = np.where((np.roll(EP, -1) == EP) & ~tr)[0]
persist_te = persist_te[persist_te < len(Z) - 1]
skill["persistence"] = float(
    1
    - ((Z[persist_te] - Ynext[persist_te]) ** 2).sum()
    / ((Ynext[persist_te] - Z[tr].mean(0)) ** 2).sum()
)
out["next_frame_skill_vs_window"] = skill

# ---- reservoir (the config the calibration picked): state, its health, and next-frame skill
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
X = np.zeros((T_STEPS, w.n), np.float32)
x = w.zeros()
for t in range(T_STEPS):
    if t > 0 and EP[t] != EP[t - 1]:
        x = w.zeros()
    x = w.step(x, U[t], DT)
    X[t] = x
validR = np.roll(EP, -1) == EP
validR[-1] = False
out["next_frame_skill_vs_window"]["reservoir(400)"] = ridge_skill(X, Ynext, validR)
out["next_frame_skill_vs_window"]["reservoir(400)+action"] = ridge_skill(
    np.hstack([X, Aoh]), Ynext, validR
)
lamX = np.linalg.eigvalsh(np.cov(X[tr].T))[::-1]
out["reservoir_state_shape"] = {
    "erank": effective_rank(lamX),
    "pr": participation_ratio(lamX),
    "k90": int(np.searchsorted(np.cumsum(lamX) / lamX.sum(), 0.90) + 1),
    "saturation_frac_abs_gt_0.9": float((np.abs(X) > 0.9).mean()),
}

# ---- B: fixed points and saddles, for constant inputs taken from real frames
rngB = np.random.default_rng(1)
pick = rngB.choice(len(U), 120, replace=False)
fps, fails, kinds = [], 0, {"stable": 0, "saddle": 0, "unstable": 0}
t0 = time.time()
for i in pick:
    fp = find_fixed_point(w, U[i], x0=np.zeros(w.n), dt=DT)
    if fp is None:
        fails += 1
        continue
    kinds[fp.kind] += 1
    fps.append(fp)
rad = np.array([f.radius for f in fps])
nun = np.array([f.n_unstable for f in fps])
out["fixed_points_from_real_frames"] = {
    "n_inputs": 120,
    "solver_failures": fails,
    "kinds": kinds,
    "radius_median": float(np.median(rad)),
    "radius_max": float(rad.max()),
    "mean_unstable_dirs": float(nun.mean()),
}
fp0 = find_fixed_point(w, np.zeros(enc.dim), dt=DT)
out["fixed_point_blank_input"] = (
    None
    if fp0 is None
    else {
        "kind": fp0.kind,
        "radius": fp0.radius,
        "n_unstable": fp0.n_unstable,
        "norm": float(np.linalg.norm(fp0.x)),
    }
)
# multistability: many random starts at one real input
starts = [find_fixed_point(w, U[pick[0]], x0=rngB.uniform(-1, 1, w.n), dt=DT) for _ in range(25)]
found = [f.x for f in starts if f is not None]
distinct = []
for xf in found:
    if all(np.linalg.norm(xf - d) > 1e-4 for d in distinct):
        distinct.append(xf)
out["multistability_25_random_starts_one_input"] = {
    "converged": len(found),
    "distinct_fixed_points": len(distinct),
}
# along the real trajectory: how often is the *local* Jacobian expanding?
loc = local_radius_along(w, U[:1500], DT, every=10)
out["along_trajectory_local_radius"] = {
    "median": float(np.median(loc)),
    "max": float(loc.max()),
    "frac_gt_1": float((loc > 1).mean()),
    "n": len(loc),
}
out["fixed_point_seconds"] = round(time.time() - t0, 1)

# ---- C: is fitness signal-to-noise adequate for the ES budget (3 episodes / candidate)?
wrapped = ReservoirObservation(env, w, enc, dt=DT)
wrapped.encoder = enc


def ep_return(theta, seed):
    pol = LinearPolicy(theta, w.n, n_act)
    o, _ = wrapped.reset(seed=seed)
    tot = 0.0
    for _ in range(2000):
        o, r, te, tru, _ = wrapped.step(pol.act(o))
        tot += r
        if te or tru:
            break
    return tot


rngC = np.random.default_rng(2)
n_theta, n_ep = 16, 12
R = np.array(
    [
        [
            ep_return(
                (0.05 * rngC.standard_normal(LinearPolicy.n_params(w.n, n_act))).astype(np.float32),
                5000 + e,
            )
            for e in range(n_ep)
        ]
        for _ in range(n_theta)
    ]
)
within = R.var(axis=1, ddof=1).mean()
between = R.mean(axis=1).var(ddof=1)
icc3 = (between - within / 3) / max(between, 1e-9)  # reliability of a 3-episode mean
out["fitness_noise"] = {
    "n_theta": n_theta,
    "n_episodes": n_ep,
    "within_sd": float(np.sqrt(within)),
    "sd_of_candidate_means": float(np.sqrt(between)),
    "reliability_3_episodes": float((between - within / 3) / between) if between > 0 else None,
    "reliability_12_episodes": float((between - within / 12) / between) if between > 0 else None,
    "candidate_mean_return_range": [float(R.mean(1).min()), float(R.mean(1).max())],
}
json.dump(out, open("docs/results/data_shape.json", "w"), indent=2)
print(json.dumps(out, indent=2))
