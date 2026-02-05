from __future__ import annotations

import torch

from difftrain.training import TextToImageBatch, train_step
from .fixtures import DummyScheduler, DummyTextEncoder, DummyUNetFixed, DummyVAE, make_generator


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
    unet = DummyUNetFixed(pred=target)

    gen_step = make_generator(123)
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
    unet = DummyUNetFixed(pred=target)

    gen_step = make_generator(456)
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
