from __future__ import annotations

import os

import torch

from difftrain.utils.repro import set_seed


def test_set_seed_sets_cublas_workspace_for_deterministic_cuda(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)

    set_seed(123, deterministic=True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_set_seed_keeps_existing_cublas_workspace_config(monkeypatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)

    set_seed(456, deterministic=True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
