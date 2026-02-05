from __future__ import annotations

from dataclasses import dataclass

import torch

from difftrain.training import compute_dream_and_update_latents


@dataclass
class DummyConfig:
    prediction_type: str


class DummyScheduler:
    def __init__(self, alphas_cumprod: torch.Tensor, prediction_type: str) -> None:
        self.alphas_cumprod = alphas_cumprod
        self.config = DummyConfig(prediction_type=prediction_type)


class DummyOutput:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class DummyUNet(torch.nn.Module):
    def __init__(self, pred: torch.Tensor) -> None:
        super().__init__()
        self._pred = pred

    def forward(self, noisy_latents, timesteps, encoder_hidden_states):  # noqa: D401
        return DummyOutput(self._pred)


def _make_latents(
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
    x0: torch.Tensor,
    noise: torch.Tensor,
):
    alphas = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    alpha_t = torch.sqrt(alphas)
    sigma_t = torch.sqrt(1.0 - alphas)
    xt = alpha_t * x0 + sigma_t * noise
    return xt, alpha_t, sigma_t


def test_dream_epsilon_prediction() -> None:
    torch.manual_seed(0)
    alphas_cumprod = torch.tensor([0.8, 0.2], dtype=torch.float32)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    x0 = torch.randn(2, 1, 2, 2)
    noise = torch.randn(2, 1, 2, 2)
    noisy_latents, alpha_t, sigma_t = _make_latents(alphas_cumprod, timesteps, x0, noise)

    target = noise
    pred_eps = noise + 0.1
    unet = DummyUNet(pred_eps)
    scheduler = DummyScheduler(alphas_cumprod, prediction_type="epsilon")

    updated_noisy, updated_target = compute_dream_and_update_latents(
        unet=unet,
        noise_scheduler=scheduler,
        timesteps=timesteps,
        noise=noise,
        noisy_latents=noisy_latents,
        target=target,
        encoder_hidden_states=torch.zeros(2, 1),
        dream_detail_preservation=1.0,
    )

    dream_lambda = sigma_t
    delta_eps = (noise - pred_eps).detach()
    expected_noisy = noisy_latents + sigma_t * (delta_eps * dream_lambda)
    expected_target = target + delta_eps * dream_lambda

    torch.testing.assert_close(updated_noisy, expected_noisy)
    torch.testing.assert_close(updated_target, expected_target)


def test_dream_v_prediction() -> None:
    torch.manual_seed(0)
    alphas_cumprod = torch.tensor([0.9, 0.3], dtype=torch.float32)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    x0 = torch.randn(2, 1, 2, 2)
    noise = torch.randn(2, 1, 2, 2)
    noisy_latents, alpha_t, sigma_t = _make_latents(alphas_cumprod, timesteps, x0, noise)

    target_v = alpha_t * noise - sigma_t * x0
    pred_v = target_v + 0.05
    unet = DummyUNet(pred_v)
    scheduler = DummyScheduler(alphas_cumprod, prediction_type="v_prediction")

    updated_noisy, updated_target = compute_dream_and_update_latents(
        unet=unet,
        noise_scheduler=scheduler,
        timesteps=timesteps,
        noise=noise,
        noisy_latents=noisy_latents,
        target=target_v,
        encoder_hidden_states=torch.zeros(2, 1),
        dream_detail_preservation=1.0,
    )

    dream_lambda = sigma_t
    pred_eps = sigma_t * noisy_latents + alpha_t * pred_v
    delta_eps = (noise - pred_eps).detach()
    expected_noisy = noisy_latents + sigma_t * (delta_eps * dream_lambda)
    expected_target = target_v + alpha_t * (delta_eps * dream_lambda)

    eps_prime = noise + delta_eps * dream_lambda
    expected_noisy_from_eps = alpha_t * x0 + sigma_t * eps_prime

    torch.testing.assert_close(updated_noisy, expected_noisy)
    torch.testing.assert_close(updated_noisy, expected_noisy_from_eps)
    torch.testing.assert_close(updated_target, expected_target)
