from __future__ import annotations

from dataclasses import dataclass

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
    def __init__(self, pred: torch.Tensor) -> None:
        super().__init__()
        self._pred = pred
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, noisy_latents, timesteps, encoder_hidden_states):  # noqa: D401
        return DummyOutput(self._pred)


def _manual_latents(vae: DummyVAE, pixel_values: torch.Tensor) -> torch.Tensor:
    return vae.encode(pixel_values).latent_dist.sample() * vae.scaling_factor


def test_train_step_epsilon_prediction() -> None:
    torch.manual_seed(0)
    alphas_cumprod = torch.tensor([0.9, 0.6, 0.3], dtype=torch.float32)
    scheduler = DummyScheduler(alphas_cumprod, prediction_type="epsilon")
    vae = DummyVAE()
    text_encoder = DummyTextEncoder()

    pixel_values = torch.randn(2, 3, 4, 4)
    input_ids = torch.randint(0, 10, (2, 4))
    batch = TextToImageBatch(pixel_values=pixel_values, input_ids=input_ids)

    gen_manual = torch.Generator().manual_seed(123)
    latents = _manual_latents(vae, pixel_values)
    timesteps = torch.randint(0, alphas_cumprod.numel(), (2,), generator=gen_manual)
    noise = torch.randn(latents.shape, generator=gen_manual)

    target = noise
    unet = DummyUNet(pred=target)

    gen_step = torch.Generator().manual_seed(123)
    output = train_step(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        noise_scheduler=scheduler,
        batch=batch,
        prediction_type="epsilon",
        dream_training=False,
        generator=gen_step,
    )

    torch.testing.assert_close(output.timesteps, timesteps)
    torch.testing.assert_close(output.target, target)
    torch.testing.assert_close(output.loss, torch.zeros_like(output.loss))


def test_train_step_v_prediction() -> None:
    torch.manual_seed(0)
    alphas_cumprod = torch.tensor([0.8, 0.5, 0.2], dtype=torch.float32)
    scheduler = DummyScheduler(alphas_cumprod, prediction_type="v_prediction")
    vae = DummyVAE()
    text_encoder = DummyTextEncoder()

    pixel_values = torch.randn(2, 3, 4, 4)
    input_ids = torch.randint(0, 10, (2, 4))
    batch = TextToImageBatch(pixel_values=pixel_values, input_ids=input_ids)

    gen_manual = torch.Generator().manual_seed(456)
    latents = _manual_latents(vae, pixel_values)
    timesteps = torch.randint(0, alphas_cumprod.numel(), (2,), generator=gen_manual)
    noise = torch.randn(latents.shape, generator=gen_manual)

    target = scheduler.get_velocity(latents, noise, timesteps)
    unet = DummyUNet(pred=target)

    gen_step = torch.Generator().manual_seed(456)
    output = train_step(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        noise_scheduler=scheduler,
        batch=batch,
        prediction_type="v_prediction",
        dream_training=False,
        generator=gen_step,
    )

    torch.testing.assert_close(output.timesteps, timesteps)
    torch.testing.assert_close(output.target, target)
    torch.testing.assert_close(output.loss, torch.zeros_like(output.loss))
