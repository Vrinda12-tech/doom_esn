"""End-to-end experiment: collect frames -> fit encoder -> calibrate -> train -> evaluate."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .calibrate import CalibrationResult, grid_search, select
from .config import ExperimentConfig
from .diagnostics import RunningMoments, StateHealth
from .encoder import FrameEncoder
from .envs import collect_pooled_frames, make_doom_env
from .es import evaluate, train_es
from .policy import LinearPolicy, RandomPolicy
from .reservoir import ReservoirConfig, ReservoirWeights
from .wrappers import ReservoirObservation

log = logging.getLogger(__name__)


def _write_calibration(path: Path, results: list[CalibrationResult]) -> None:
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["rho", "input_scaling", "tau", "lyapunov", "horizon", "erank", "pr", "cond"])
        for r in results:
            c, h = r.cfg, r.health
            wr.writerow(
                [
                    c.spectral_radius,
                    c.input_scaling,
                    c.tau,
                    round(r.lyapunov, 4),
                    round(r.horizon, 2),
                    round(h.erank, 2),
                    round(h.pr, 2),
                    f"{h.cond:.3g}",
                ]
            )


def save_checkpoint(
    path: Path, theta: Any, cfg: ExperimentConfig, res_cfg: ReservoirConfig, enc: FrameEncoder
) -> None:
    np.savez(
        path,
        theta=theta,
        res_cfg=json.dumps(asdict(res_cfg)),
        exp_cfg=json.dumps(asdict(cfg)),
        version=__version__,
        **enc.to_arrays(),
    )


def load_checkpoint(path: Path) -> tuple[Any, ReservoirConfig, ExperimentConfig, FrameEncoder]:
    from .config import build

    z = np.load(path, allow_pickle=False)
    exp = build(ExperimentConfig, json.loads(str(z["exp_cfg"])))
    res = ReservoirConfig(**json.loads(str(z["res_cfg"])))
    enc = FrameEncoder.from_arrays(exp.encoder.height, exp.encoder.width, z)
    return z["theta"], res, exp, enc


def run_experiment(cfg: ExperimentConfig, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    env = make_doom_env(cfg.env.env_id, cfg.env.frame_skip)
    enc = FrameEncoder(cfg.encoder.height, cfg.encoder.width)
    dt = float(cfg.env.frame_skip)

    log.info("collecting %d random-policy frames", cfg.env.calib_steps)
    pooled = collect_pooled_frames(env, enc, cfg.env.calib_steps, seed=cfg.es.seed)
    enc.fit(pooled)
    inputs = enc.transform_pooled(pooled)

    res_cfg = cfg.reservoir
    calib = cfg.calibration
    if calib.enabled:
        results = grid_search(
            res_cfg, inputs, dt, calib.spectral_radii, calib.input_scalings, calib.taus, calib.burn
        )
        _write_calibration(out_dir / "calibration.csv", results)
        chosen = select(results, calib.max_lyapunov, calib.min_horizon, calib.max_horizon)
        res_cfg = chosen.cfg
        log.info(
            "selected %s  erank=%.1f  lyapunov=%.3f", res_cfg, chosen.health.erank, chosen.lyapunov
        )

    weights = ReservoirWeights.build(res_cfg, enc.dim)
    wrapped = ReservoirObservation(env, weights, enc, dt=dt)
    es_res = train_es(wrapped, cfg.es, out_dir)

    n_feat, n_act = weights.n, int(env.action_space.n)
    trained = LinearPolicy(es_res.theta, n_feat, n_act)
    held_out = 10**7  # seeds disjoint from training
    moments = RunningMoments(n_feat)
    wrapped.moments = moments
    r_trained = evaluate(wrapped, trained, cfg.es.eval_episodes, held_out, cfg.es.max_steps)
    wrapped.moments = None
    r_random = evaluate(
        wrapped, RandomPolicy(n_act, cfg.es.seed), cfg.es.eval_episodes, held_out, cfg.es.max_steps
    )
    save_checkpoint(out_dir / "checkpoint.npz", es_res.theta, cfg, res_cfg, enc)

    def stats(r: Any) -> dict[str, float]:
        return {"mean": float(np.mean(r)), "sem": float(np.std(r) / np.sqrt(len(r)))}

    report = {
        "reservoir": asdict(res_cfg),
        "reservoir_bytes": weights.nbytes(),
        "n_policy_params": LinearPolicy.n_params(n_feat, n_act),
        "generations": es_res.generations,
        "heldout_trained": stats(r_trained),
        "heldout_random": stats(r_random),
        "eval_state_health": asdict(StateHealth.from_moments(moments)),
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    env.close()
    return report
