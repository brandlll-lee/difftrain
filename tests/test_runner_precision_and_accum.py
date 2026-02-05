from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from difftrain.config import ExperimentConfig
from difftrain.training import run_text_to_image_training, TextToImageBatch

from .fixtures import DummyScheduler, DummyTextEncoder, DummyUNetTrain, DummyVAE


class DummyDataset(Dataset):
    def __init__(self, num_items: int = 4) -> None:
        self.num_items = num_items

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int) -> TextToImageBatch:
        pixel_values = torch.randn(1, 3, 4, 4)
        input_ids = torch.randint(0, 10, (1, 4))
        return TextToImageBatch(pixel_values=pixel_values, input_ids=input_ids)


class CountingOptimizer(torch.optim.Optimizer):
    def __init__(self, params) -> None:
        defaults = {"lr": 1e-3}
        super().__init__(params, defaults)
        self.step_calls = 0

    def step(self, closure=None):  # type: ignore[override]
        self.step_calls += 1
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.detach_()
                        param.grad.zero_()


def _make_runner_components():
    alphas_cumprod = torch.tensor([0.9, 0.6, 0.3], dtype=torch.float32)
    scheduler = DummyScheduler(alphas_cumprod, prediction_type="epsilon")
    vae = DummyVAE()
    text_encoder = DummyTextEncoder()
    unet = DummyUNetTrain()
    return scheduler, vae, text_encoder, unet


def test_runner_precision_fallback_cpu(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.train.precision = "fp16"
    cfg.train.max_steps = 1
    cfg.train.grad_accum_steps = 1
    cfg.train.checkpointing_steps = 1
    cfg.train.log_every = 1
    cfg.model.prediction_type = "epsilon"

    scheduler, vae, text_encoder, unet = _make_runner_components()
    optimizer = CountingOptimizer(unet.parameters())
    dataloader = DataLoader(DummyDataset(num_items=1), batch_size=1, collate_fn=lambda x: x[0])

    state = run_text_to_image_training(
        cfg=cfg,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        noise_scheduler=scheduler,
        optimizer=optimizer,
        train_dataloader=dataloader,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert state.global_step == 1
    assert cfg.train.precision == "fp32"


def test_runner_grad_accum_counts_optimizer_steps(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.train.max_steps = 2
    cfg.train.grad_accum_steps = 2
    cfg.train.checkpointing_steps = 1
    cfg.train.log_every = 1
    cfg.model.prediction_type = "epsilon"

    scheduler, vae, text_encoder, unet = _make_runner_components()
    optimizer = CountingOptimizer(unet.parameters())
    dataloader = DataLoader(DummyDataset(num_items=4), batch_size=1, collate_fn=lambda x: x[0])

    state = run_text_to_image_training(
        cfg=cfg,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        noise_scheduler=scheduler,
        optimizer=optimizer,
        train_dataloader=dataloader,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert state.global_step == 2
    assert optimizer.step_calls == 2


def test_checkpoint_payload_fields_present(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.train.max_steps = 1
    cfg.train.grad_accum_steps = 1
    cfg.train.checkpointing_steps = 1
    cfg.train.log_every = 1
    cfg.model.prediction_type = "epsilon"

    scheduler, vae, text_encoder, unet = _make_runner_components()
    optimizer = CountingOptimizer(unet.parameters())
    dataloader = DataLoader(DummyDataset(num_items=1), batch_size=1, collate_fn=lambda x: x[0])

    run_text_to_image_training(
        cfg=cfg,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        noise_scheduler=scheduler,
        optimizer=optimizer,
        train_dataloader=dataloader,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    checkpoint_path = tmp_path / "checkpoint-1.pt"
    assert checkpoint_path.exists()
    payload = torch.load(checkpoint_path, map_location="cpu")

    assert "global_step" in payload
    assert "unet" in payload
    assert "optimizer" in payload
    assert "precision" in payload
    assert "scaler" in payload
    assert "config" in payload
