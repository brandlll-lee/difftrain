from __future__ import annotations

from typing import Optional

import pytest
import torch


def require_text2image_deps() -> None:
    """Skip test if text2image optional dependencies are missing."""
    pytest.importorskip("datasets")
    pytest.importorskip("diffusers")
    pytest.importorskip("PIL")
    pytest.importorskip("torchvision")
    pytest.importorskip("transformers")


def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    msg: Optional[str] = None,
) -> None:
    """Assert tensors are close with a concise error message.

    Args:
        actual: Actual tensor.
        expected: Expected tensor.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        msg: Optional message prefix.
    """
    if actual.shape != expected.shape:
        prefix = f"{msg} " if msg else ""
        raise AssertionError(f"{prefix}Shape mismatch: actual {actual.shape}, expected {expected.shape}")

    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        diff = (actual - expected).abs()
        max_diff = float(diff.max().item()) if diff.numel() else 0.0
        mean_diff = float(diff.mean().item()) if diff.numel() else 0.0
        prefix = f"{msg} " if msg else ""
        raise AssertionError(
            f"{prefix}Tensors not close (max_diff={max_diff:.6g}, mean_diff={mean_diff:.6g}, "
            f"atol={atol}, rtol={rtol})"
        ) from exc
