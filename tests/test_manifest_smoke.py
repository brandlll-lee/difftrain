from __future__ import annotations

import json
from pathlib import Path

from difftrain.config import ExperimentConfig, save_config
from difftrain.utils.manifest import write_run_manifest
from difftrain.utils.repro import resolve_output_dir, set_seed, write_env_report


def test_run_manifest_written(tmp_path: Path) -> None:
    out_dir = resolve_output_dir(tmp_path / "run")

    cfg = ExperimentConfig()
    cfg_path = out_dir / "config.yaml"
    save_config(cfg, cfg_path)

    set_seed(cfg.train.seed, deterministic=True)
    env_path = write_env_report(out_dir / "env.json")

    resolved_path = out_dir / "config.resolved.yaml"
    save_config(cfg, resolved_path)

    manifest_path = write_run_manifest(
        output_dir=out_dir,
        config_path=cfg_path,
        cfg=cfg,
        deterministic=True,
        env_json_path=env_path,
        resolved_config_path=resolved_path,
        repo_dir=tmp_path,  # not a git repo
    )

    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["paths"]["output_dir"]
    assert payload["env"]["torch_version"]
    assert payload["config"]["source"]["sha256"]
    assert payload["data_fingerprint"]["status"] == "placeholder"

