from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import ExperimentConfig

MANIFEST_SCHEMA_VERSION = 1


def _sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_pkg_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def collect_dependency_versions() -> Dict[str, Optional[str]]:
    """
    Collect versions of key runtime dependencies.
    Keep this intentionally small and stable (do not dump full `pip freeze` by default).
    """
    return {
        "python": sys.version.replace("\n", " "),
        "difftrain": _safe_pkg_version("difftrain"),
        "torch": _safe_pkg_version("torch"),
        "numpy": _safe_pkg_version("numpy"),
        "pyyaml": _safe_pkg_version("pyyaml"),
        "pip": _safe_pkg_version("pip"),
    }


def _run_git(repo_dir: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_dir),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def collect_git_info(repo_dir: Path) -> Dict[str, Any]:
    """
    Best-effort git info. Works even when not a git repo.
    """
    repo_dir = repo_dir.resolve()
    info: Dict[str, Any] = {
        "available": False,
        "repo_dir": str(repo_dir),
        "is_repo": False,
        "commit": None,
        "branch": None,
        "dirty": None,
        "describe": None,
        "error": None,
    }

    # Fast path: if .git doesn't exist, don't spend time.
    if not (repo_dir / ".git").exists():
        info["error"] = "no .git directory found"
        return info

    try:
        p = _run_git(repo_dir, ["rev-parse", "--is-inside-work-tree"])
        if p.returncode != 0 or p.stdout.strip() != "true":
            info["error"] = (p.stderr or p.stdout).strip() or "not a git worktree"
            return info
        info["available"] = True
        info["is_repo"] = True

        commit = _run_git(repo_dir, ["rev-parse", "HEAD"]).stdout.strip()
        branch = _run_git(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
        describe = _run_git(repo_dir, ["describe", "--always", "--dirty", "--tags"]).stdout.strip()
        status = _run_git(repo_dir, ["status", "--porcelain"]).stdout

        info["commit"] = commit or None
        info["branch"] = branch or None
        info["describe"] = describe or None
        info["dirty"] = bool(status.strip())
        return info
    except FileNotFoundError:
        info["error"] = "git executable not found"
        return info
    except Exception as e:
        info["error"] = f"git info collection failed: {type(e).__name__}: {e}"
        return info


def collect_data_fingerprint_placeholder(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Placeholder for future data engine fingerprinting.
    """
    return {
        "status": "placeholder",
        "method": None,
        "value": None,
        "inputs": {
            "dataset_type": cfg.data.dataset_type,
            "data_root": cfg.data.data_root,
            "image_size": cfg.data.image_size,
        },
        "notes": (
            "Data pipeline is not implemented yet. "
            "Replace with dataset version/hash + sampling rules later."
        ),
    }


def write_run_manifest(
    *,
    output_dir: Path,
    config_path: Path,
    cfg: ExperimentConfig,
    deterministic: bool,
    env_json_path: Path,
    resolved_config_path: Path,
    manifest_path: Optional[Path] = None,
    repo_dir: Optional[Path] = None,
) -> Path:
    """
    Write a single JSON manifest that captures:
    - config (resolved)
    - env snapshot
    - git info (best-effort)
    - key dependency versions
    - data fingerprint placeholder
    """
    output_dir = output_dir.resolve()
    config_path = config_path.resolve()
    env_json_path = env_json_path.resolve()
    resolved_config_path = resolved_config_path.resolve()
    repo_dir = (repo_dir or output_dir).resolve()

    if manifest_path is None:
        manifest_path = output_dir / "run_manifest.json"
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Load env snapshot written by `write_env_report` (single source of truth).
    env_dict = json.loads(env_json_path.read_text(encoding="utf-8"))

    # Config: keep both a structured dict and a checksum of the source config file.
    config_source_text = config_path.read_text(encoding="utf-8")
    cfg_dict = asdict(cfg)

    created_utc = datetime.now(timezone.utc).isoformat()
    payload: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": created_utc,
        "paths": {
            "output_dir": str(output_dir),
            "config_path": str(config_path),
            "env_json": str(env_json_path),
            "config_resolved_yaml": str(resolved_config_path),
            "run_manifest_json": str(manifest_path),
        },
        "artifacts": {
            "env_json": env_json_path.name,
            "config_resolved_yaml": resolved_config_path.name,
            "run_manifest_json": manifest_path.name,
        },
        "repro": {
            "seed": cfg.train.seed,
            "deterministic": bool(deterministic),
        },
        "git": collect_git_info(repo_dir),
        "dependencies": collect_dependency_versions(),
        "config": {
            "source": {
                "path": str(config_path),
                "sha256": _sha256_text(config_source_text),
            },
            "resolved": cfg_dict,
            "resolved_file_sha256": (
                _sha256_file(resolved_config_path) if resolved_config_path.exists() else None
            ),
        },
        "env": env_dict,
        "data_fingerprint": collect_data_fingerprint_placeholder(cfg),
    }

    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path

