from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke check reproducibility artifacts exist.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./difftrain/_tmp_run",
        help="Output directory to inspect.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    required = [
        out_dir / "env.json",
        out_dir / "config.resolved.yaml",
        out_dir / "run_manifest.json",
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"[difftrain] missing artifacts: {[str(p) for p in missing]}")

    print("[difftrain] smoke_repro: OK")
    for p in required:
        print(f"[difftrain] found: {p}")


if __name__ == "__main__":
    main()

