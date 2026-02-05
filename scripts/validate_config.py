from __future__ import annotations

import argparse
from pathlib import Path

from difftrain.config import load_config, save_config
from difftrain.utils.manifest import write_run_manifest
from difftrain.utils.repro import resolve_output_dir, set_seed, write_env_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and snapshot resolved config for reproducibility.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output dir (else use config.train.output_dir).",
    )
    parser.add_argument("--deterministic", action="store_true", help="Enable torch deterministic algorithms.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = resolve_output_dir(args.output_dir or cfg.train.output_dir)

    # Reproducibility snapshots
    set_seed(cfg.train.seed, deterministic=args.deterministic)
    env_path = write_env_report(out_dir / "env.json")

    # Persist the fully-resolved config used for the run
    resolved_cfg_path = out_dir / "config.resolved.yaml"
    save_config(cfg, resolved_cfg_path)

    manifest_path = write_run_manifest(
        output_dir=out_dir,
        config_path=Path(args.config),
        cfg=cfg,
        deterministic=args.deterministic,
        env_json_path=env_path,
        resolved_config_path=resolved_cfg_path,
        repo_dir=Path.cwd(),
    )

    print(f"[difftrain] config: {Path(args.config).resolve()}")
    print(f"[difftrain] output_dir: {out_dir.resolve()}")
    print(f"[difftrain] wrote: {resolved_cfg_path.resolve()}")
    print(f"[difftrain] wrote: {(out_dir / 'env.json').resolve()}")
    print(f"[difftrain] wrote: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()

