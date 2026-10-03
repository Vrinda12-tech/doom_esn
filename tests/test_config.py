from __future__ import annotations

import pytest

from doom_esn.config import ConfigError, ExperimentConfig, build, load_config


def test_unknown_key_is_an_error() -> None:
    with pytest.raises(ConfigError):
        build(ExperimentConfig, {"es": {"popsise": 4}})


def test_shipped_config_loads() -> None:
    cfg = load_config("configs/basic.yaml")
    assert cfg.reservoir.n_res == 400
    assert cfg.calibration.taus == (4.0, 8.0, 16.0)
