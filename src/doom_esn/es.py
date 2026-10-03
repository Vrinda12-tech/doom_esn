"""Gradient-free training of the linear readout with diagonal CMA-ES (pycma)."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ._types import F32
from .config import ESConfig
from .diagnostics import RunningMoments, StateHealth
from .policy import LinearPolicy
from .wrappers import FeatureObservation

log = logging.getLogger(__name__)


class _Policy(Protocol):
    def act(self, x: F32) -> int: ...


def rollout(env: FeatureObservation, policy: _Policy, seed: int, max_steps: int) -> float:
    obs, _ = env.reset(seed=seed)
    total = 0.0
    for _ in range(max_steps):
        obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
        total += reward
        if terminated or truncated:
            break
    return total


def evaluate(
    env: FeatureObservation, policy: _Policy, episodes: int, seed0: int, max_steps: int
) -> F32:
    return np.array(
        [rollout(env, policy, seed0 + i, max_steps) for i in range(episodes)], dtype=np.float32
    )


@dataclass(frozen=True)
class ESResult:
    theta: F32
    generations: int
    best_mean_return: float
    final_health: StateHealth | None


def train_es(env: FeatureObservation, cfg: ESConfig, out_dir: Path) -> ESResult:
    import cma

    n_feat = int(env.observation_space.shape[0])  # type: ignore[index]
    n_act = int(env.action_space.n)  # type: ignore[attr-defined]
    dim = LinearPolicy.n_params(n_feat, n_act)
    opts: dict[str, Any] = {
        "CMA_diagonal": True,
        "popsize": cfg.popsize,
        "seed": cfg.seed + 1,
        "verbose": -9,
    }
    es = cma.CMAEvolutionStrategy(np.zeros(dim), cfg.sigma0, opts)
    moments = RunningMoments(env.memory_dim) if cfg.track_health else None
    env.moments = moments
    t0, best, gen = time.monotonic(), -np.inf, 0
    health: StateHealth | None = None
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["gen", "best_return", "mean_return", "erank", "pr", "cond", "seconds"])
        while gen < cfg.generations and not es.stop():
            sols = es.ask()
            rets = []
            for s in sols:
                pol = LinearPolicy(np.asarray(s, dtype=np.float32), n_feat, n_act)
                rets.append(
                    float(
                        np.mean(
                            [
                                rollout(env, pol, cfg.seed * 10**6 + gen * 1000 + e, cfg.max_steps)
                                for e in range(cfg.episodes)
                            ]
                        )
                    )
                )
            es.tell(sols, [-r for r in rets])
            best = max(best, max(rets))
            health = (
                StateHealth.from_moments(moments)
                if moments is not None and moments.count > 1
                else None
            )
            wr.writerow(
                [
                    gen,
                    max(rets),
                    float(np.mean(rets)),
                    health.erank if health else "",
                    health.pr if health else "",
                    health.cond if health else "",
                    round(time.monotonic() - t0, 1),
                ]
            )
            fh.flush()
            log.info("gen %d  best %.1f  mean %.1f", gen, max(rets), float(np.mean(rets)))
            gen += 1
            if cfg.max_seconds is not None and time.monotonic() - t0 > cfg.max_seconds:
                log.warning("stopping early: time budget reached")
                break
    env.moments = None
    theta = np.asarray(es.result.xfavorite, dtype=np.float32)
    return ESResult(theta, gen, float(best), health)
