from __future__ import annotations

from dataclasses import dataclass

import torch


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


class DummyUNetFixed(torch.nn.Module):
    def __init__(self, pred: torch.Tensor) -> None:
        super().__init__()
        self._pred = pred
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, noisy_latents, timesteps, encoder_hidden_states):  # noqa: D401
        return DummyOutput(self._pred)


class DummyUNetTrain(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, noisy_latents, timesteps, encoder_hidden_states):  # noqa: D401
        return DummyOutput(noisy_latents + self.dummy)


def make_alphas(num_steps: int, *, start: float = 0.9, end: float = 0.1) -> torch.Tensor:
    """Create a simple linear alphas_cumprod tensor."""
    return torch.linspace(start, end, steps=num_steps, dtype=torch.float32)


def make_generator(seed: int) -> torch.Generator:
    """Create a torch.Generator with a fixed seed."""
    return torch.Generator().manual_seed(seed)
