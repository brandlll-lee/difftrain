from __future__ import annotations

import json
from pathlib import Path

from difftrain.config import ExperimentConfig, save_config
from difftrain.utils.manifest import write_run_manifest
from difftrain.utils.repro import resolve_output_dir, set_seed, write_env_report


def test_repro_artifacts_written(tmp_path: Path) -> None:
    out_dir = resolve_output_dir(tmp_path / "run")
    cfg = ExperimentConfig()

    cfg_path = out_dir / "config.yaml"
    save_config(cfg, cfg_path)

    set_seed(cfg.train.seed, deterministic=True)
    env_path = write_env_report(out_dir / "env.json")

    resolved_cfg_path = out_dir / "config.resolved.yaml"
    save_config(cfg, resolved_cfg_path)

    manifest_path = write_run_manifest(
        output_dir=out_dir,
        config_path=cfg_path,
        cfg=cfg,
        deterministic=True,
        env_json_path=env_path,
        resolved_config_path=resolved_cfg_path,
        repo_dir=tmp_path,
    )

    assert env_path.exists()
    assert resolved_cfg_path.exists()
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["repro"]["seed"] == cfg.train.seed
    assert payload["paths"]["config_resolved_yaml"].endswith("config.resolved.yaml")
