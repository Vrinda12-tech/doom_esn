"""Command line: ``doom-esn train|eval``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .config import load_config
from .envs import make_doom_env
from .es import evaluate
from .pipeline import load_checkpoint, run_experiment
from .policy import LinearPolicy
from .reservoir import ReservoirWeights
from .wrappers import ReservoirObservation


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="doom-esn")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="calibrate the reservoir, train the readout, evaluate")
    t.add_argument("--config", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("eval", help="evaluate a saved checkpoint")
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--episodes", type=int, default=20)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "train":
        report = run_experiment(load_config(args.config), args.out)
        print(json.dumps(report, indent=2))
        return 0

    theta, res_cfg, exp, enc = load_checkpoint(args.checkpoint)
    env = make_doom_env(exp.env.env_id, exp.env.frame_skip)
    w = ReservoirWeights.build(res_cfg, enc.dim)
    wrapped = ReservoirObservation(env, w, enc, dt=float(exp.env.frame_skip))
    pol = LinearPolicy(theta, w.n, int(env.action_space.n))
    r = evaluate(wrapped, pol, args.episodes, 10**7, exp.es.max_steps)
    print(f"return {np.mean(r):.1f} +/- {np.std(r) / np.sqrt(len(r)):.1f} over {len(r)} episodes")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
