from __future__ import annotations

import json
import os
import platform
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """
    Set seeds for Python, NumPy and PyTorch.

    Notes:
    - Determinism in GPU kernels may require additional env vars and can reduce performance.
    - Some ops remain nondeterministic depending on backend/hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # PyTorch determinism knobs
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Raises if a nondeterministic op is used (helps catch issues early).
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older torch versions might not support this fully.
            pass


@dataclass(frozen=True)
class EnvReport:
    created_utc: str
    python: str
    platform: str
    executable: str
    argv: list[str]
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    cudnn_version: Optional[int]
    device_count: int
    devices: list[Dict[str, Any]]

    @staticmethod
    def collect() -> "EnvReport":
        created_utc = datetime.now(timezone.utc).isoformat()
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if hasattr(torch.version, "cuda") else None
        cudnn_version = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None

        devices: list[Dict[str, Any]] = []
        device_count = torch.cuda.device_count() if cuda_available else 0
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "major": int(getattr(props, "major", -1)),
                    "minor": int(getattr(props, "minor", -1)),
                    "multi_processor_count": int(getattr(props, "multi_processor_count", -1)),
                }
            )

        return EnvReport(
            created_utc=created_utc,
            python=sys.version.replace("\n", " "),
            platform=platform.platform(),
            executable=sys.executable,
            argv=list(sys.argv),
            torch_version=str(torch_version),
            cuda_available=bool(cuda_available),
            cuda_version=str(cuda_version) if cuda_version is not None else None,
            cudnn_version=int(cudnn_version) if cudnn_version is not None else None,
            device_count=int(device_count),
            devices=devices,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_utc": self.created_utc,
            "python": self.python,
            "platform": self.platform,
            "executable": self.executable,
            "argv": self.argv,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "device_count": self.device_count,
            "devices": self.devices,
        }


def write_env_report(path: str | Path) -> Path:
    """
    Collect and write an environment report to JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = EnvReport.collect()
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def resolve_output_dir(output_dir: str | Path) -> Path:
    """
    Resolve and create output dir. Does not enforce any specific run naming yet.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out

