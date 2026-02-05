from __future__ import annotations

from pathlib import Path

from difftrain.config import ExperimentConfig, load_config, save_config


def test_config_roundtrip(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    path = tmp_path / "config.yaml"
    save_config(cfg, path)

    cfg2 = load_config(path)
    # Compare a few key fields (dataclass equality also works but keep it explicit)
    assert cfg2.train.seed == cfg.train.seed
    assert cfg2.train.output_dir == cfg.train.output_dir
    assert cfg2.model.prediction_type == cfg.model.prediction_type
    assert cfg2.scheduler.num_train_timesteps == cfg.scheduler.num_train_timesteps

