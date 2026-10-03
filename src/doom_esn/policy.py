"""Linear readout policy: a = argmax(Theta [x; 1])."""

from __future__ import annotations

import numpy as np

from ._types import F32


class LinearPolicy:
    def __init__(self, theta: F32, n_features: int, n_actions: int) -> None:
        if theta.size != self.n_params(n_features, n_actions):
            raise ValueError("theta has the wrong size")
        self.n_features, self.n_actions = n_features, n_actions
        self.theta = theta.reshape(n_actions, n_features + 1).astype(np.float32)

    @staticmethod
    def n_params(n_features: int, n_actions: int) -> int:
        return n_actions * (n_features + 1)

    def logits(self, x: F32) -> F32:
        return (self.theta[:, :-1] @ x + self.theta[:, -1]).astype(np.float32)

    def act(self, x: F32) -> int:
        return int(np.argmax(self.logits(x)))


class RandomPolicy:
    def __init__(self, n_actions: int, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def act(self, x: F32) -> int:
        return int(self._rng.integers(self.n_actions))
