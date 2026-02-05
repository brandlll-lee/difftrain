from __future__ import annotations

import sys
import warnings
from os.path import abspath, dirname, join
from typing import List

import pytest


# Allow running tests without installing the package in editable mode.
_repo_src_path = abspath(join(dirname(dirname(__file__)), "src"))
if _repo_src_path not in sys.path:
    sys.path.insert(1, _repo_src_path)

# Silence noisy warnings in tests; we keep actionable warnings visible.
warnings.simplefilter(action="ignore", category=FutureWarning)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: fast unit tests (CPU, offline)")
    config.addinivalue_line("markers", "integration: integration tests (may use external deps)")
    config.addinivalue_line("markers", "slow: slow tests (may download models or run training)")
    config.addinivalue_line("markers", "cuda: tests that require CUDA")
    config.addinivalue_line("markers", "text2image: tests that require text2image extras")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (may require extra deps).",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (may download models or run training).",
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    run_integration = config.getoption("--run-integration")
    run_slow = config.getoption("--run-slow")

    skip_integration = pytest.mark.skip(reason="integration tests are disabled (use --run-integration)")
    skip_slow = pytest.mark.skip(reason="slow tests are disabled (use --run-slow)")
    skip_cuda = pytest.mark.skip(reason="CUDA is not available on this machine")

    for item in items:
        if "integration" in item.keywords and not run_integration:
            item.add_marker(skip_integration)
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "cuda" in item.keywords and not _cuda_available():
            item.add_marker(skip_cuda)
