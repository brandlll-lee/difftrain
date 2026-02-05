from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from difftrain.config import ExperimentConfig
from difftrain.training import TextToImageBatch, run_text_to_image_training


@dataclass
class DummyConfig:
    prediction_type: str
    num_train_timesteps: int


class DummyScheduler:
    def __init__(self, alphas_cumprod: torch.Tensor, prediction_type: str) -> None:
        self.alphas_cumprod = alphas_cumprod
        self.config = DummyConfig(prediction_type=prediction_type, num_train_timesteps=alphas_cumprod.numel())

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alphas = self.alphas_cumprod.to(x0.device)[timesteps].view(-1, 1, 1, 1)
        alpha_t = torch.sqrt(alphas)
        sigma_t = torch.sqrt(1.0 - alphas)
        return alpha_t * x0 + sigma_t * noise

    def get_velocity(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alphas = self.alphas_cumprod.to(x0.device)[timesteps].view(-1, 1, 1, 1)
        alpha_t = torch.sqrt(alphas)
        sigma_t = torch.sqrt(1.0 - alphas)
        return alpha_t * noise - sigma_t * x0


class DummyLatentDist:
    def __init__(self, latents: torch.Tensor) -> None:
        self._latents = latents

    def sample(self) -> torch.Tensor:
        return self._latents


class DummyEncodeOut:
    def __init__(self, latents: torch.Tensor) -> None:
        self.latent_dist = DummyLatentDist(latents)


class DummyVAE:
    def __init__(self, scaling_factor: float = 0.5) -> None:
        self.scaling_factor = scaling_factor

    def encode(self, pixel_values: torch.Tensor) -> DummyEncodeOut:
        return DummyEncodeOut(pixel_values * 0.25)


class DummyTextEncoder:
    def __call__(self, input_ids: torch.Tensor):
        hidden = input_ids.float().unsqueeze(-1)
        return (hidden,)


class DummyOutput:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class DummyUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, noisy_latents, timesteps, encoder_hidden_states):  # noqa: D401
        return DummyOutput(noisy_latents + self.dummy)


class DummyDataset(Dataset):
    def __init__(self, num_items: int = 4) -> None:
        self.num_items = num_items

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int) -> TextToImageBatch:
        pixel_values = torch.randn(1, 3, 4, 4)
        input_ids = torch.randint(0, 10, (1, 4))
        return TextToImageBatch(pixel_values=pixel_values, input_ids=input_ids)


def test_runner_saves_checkpoints(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.train.max_steps = 2
    cfg.train.grad_accum_steps = 1
    cfg.train.checkpointing_steps = 1
    cfg.train.log_every = 1
    cfg.model.prediction_type = "epsilon"

    alphas_cumprod = torch.tensor([0.9, 0.6, 0.3], dtype=torch.float32)
    scheduler = DummyScheduler(alphas_cumprod, prediction_type="epsilon")
    vae = DummyVAE()
    text_encoder = DummyTextEncoder()
    unet = DummyUNet()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    dataset = DummyDataset(num_items=2)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])

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

    checkpoints = list(tmp_path.glob("checkpoint-*.pt"))
    assert state.global_step == 2
    assert len(checkpoints) >= 1


def test_runner_resume_from_checkpoint(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.train.max_steps = 1
    cfg.train.grad_accum_steps = 1
    cfg.train.checkpointing_steps = 1
    cfg.train.log_every = 1
    cfg.model.prediction_type = "epsilon"

    alphas_cumprod = torch.tensor([0.9, 0.6, 0.3], dtype=torch.float32)
    scheduler = DummyScheduler(alphas_cumprod, prediction_type="epsilon")
    vae = DummyVAE()
    text_encoder = DummyTextEncoder()
    unet = DummyUNet()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    dataset = DummyDataset(num_items=2)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])

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

    ckpt_path = tmp_path / "checkpoint-1.pt"
    assert ckpt_path.exists()

    cfg_resume = ExperimentConfig()
    cfg_resume.train.max_steps = 2
    cfg_resume.train.grad_accum_steps = 1
    cfg_resume.train.checkpointing_steps = 1
    cfg_resume.train.log_every = 1
    cfg_resume.train.resume_from = str(ckpt_path)
    cfg_resume.model.prediction_type = "epsilon"

    unet2 = DummyUNet()
    optimizer2 = torch.optim.AdamW(unet2.parameters(), lr=1e-4)

    state = run_text_to_image_training(
        cfg=cfg_resume,
        unet=unet2,
        vae=vae,
        text_encoder=text_encoder,
        noise_scheduler=scheduler,
        optimizer=optimizer2,
        train_dataloader=dataloader,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert state.global_step == 2
    assert (tmp_path / "checkpoint-2.pt").exists()
