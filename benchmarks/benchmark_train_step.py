from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import torch

from difftrain.training import TextToImageBatch, train_step


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
    def forward(self, noisy_latents, timesteps, encoder_hidden_states):  # noqa: D401
        return DummyOutput(noisy_latents)


def benchmark_train_step(iters: int = 50) -> Dict[str, float]:
    """Run a tiny CPU benchmark for train_step.

    Args:
        iters: Number of iterations to run.

    Returns:
        Dictionary with avg latency (ms) and steps/sec.
    """
    alphas_cumprod = torch.linspace(0.9, 0.1, steps=100)
    scheduler = DummyScheduler(alphas_cumprod, prediction_type="epsilon")
    vae = DummyVAE()
    text_encoder = DummyTextEncoder()
    unet = DummyUNet()

    pixel_values = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, 10, (2, 16))
    batch = TextToImageBatch(pixel_values=pixel_values, input_ids=input_ids)

    start = time.perf_counter()
    for _ in range(iters):
        _ = train_step(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            noise_scheduler=scheduler,
            batch=batch,
            prediction_type="epsilon",
            dream_training=False,
        )
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / iters) * 1000.0
    return {"avg_ms": avg_ms, "steps_per_sec": iters / elapsed}


if __name__ == "__main__":
    metrics = benchmark_train_step()
    print(metrics)
