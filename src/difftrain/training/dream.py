from __future__ import annotations

from typing import Optional, Tuple

import torch


def _get_prediction_type(noise_scheduler: object) -> str:
    config = getattr(noise_scheduler, "config", None)
    if config is not None and hasattr(config, "prediction_type"):
        return str(config.prediction_type)
    if hasattr(noise_scheduler, "prediction_type"):
        return str(noise_scheduler.prediction_type)
    raise AttributeError("noise_scheduler must expose prediction_type or config.prediction_type")


def _extract_alphas_cumprod(noise_scheduler: object, timesteps: torch.Tensor) -> torch.Tensor:
    if not hasattr(noise_scheduler, "alphas_cumprod"):
        raise AttributeError("noise_scheduler must expose alphas_cumprod")
    alphas_cumprod = noise_scheduler.alphas_cumprod
    if not isinstance(alphas_cumprod, torch.Tensor):
        alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
    alphas_cumprod = alphas_cumprod.to(device=timesteps.device)
    return alphas_cumprod[timesteps]


def _expand_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.dim() == ref.dim():
        return x
    # Expand [B] -> [B, 1, 1, 1] (or other dims) to match ref
    return x.view(-1, *([1] * (ref.dim() - 1)))


def _unwrap_prediction(pred: object) -> torch.Tensor:
    if isinstance(pred, torch.Tensor):
        return pred
    if isinstance(pred, tuple) and pred:
        return pred[0]
    if hasattr(pred, "sample"):
        return pred.sample
    raise TypeError("UNet output must be a Tensor or have a .sample attribute")


def compute_dream_and_update_latents(
    unet: torch.nn.Module,
    noise_scheduler: object,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    target: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    dream_detail_preservation: float = 1.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Apply DREAM (Diffusion Rectification and Estimation-Adaptive Models).

    Args:
        unet: Conditional UNet used for prediction (no gradients).
        noise_scheduler: Noise scheduler with `alphas_cumprod` and `prediction_type`.
        timesteps: Timesteps tensor.
        noise: Sampled noise tensor.
        noisy_latents: Noisy latents xt.
        target: Original target tensor.
        encoder_hidden_states: Conditioning states from text encoder.
        dream_detail_preservation: Detail preservation factor p.

    Returns:
        Tuple of (updated_noisy_latents, updated_target).

    Raises:
        ValueError: If prediction_type is unsupported.
    """
    alphas_cumprod = _extract_alphas_cumprod(noise_scheduler, timesteps)
    alphas_cumprod = _expand_like(alphas_cumprod, noisy_latents)
    alpha_t = torch.sqrt(alphas_cumprod)
    sigma_t = torch.sqrt(1.0 - alphas_cumprod)

    dream_lambda = sigma_t**dream_detail_preservation

    with torch.no_grad():
        pred = _unwrap_prediction(unet(noisy_latents, timesteps, encoder_hidden_states))

    prediction_type = _get_prediction_type(noise_scheduler)
    if prediction_type == "epsilon":
        pred_eps = pred
        delta_eps = (noise - pred_eps).detach()
        delta_eps.mul_(dream_lambda)
        updated_noisy_latents = noisy_latents + sigma_t * delta_eps
        updated_target = target + delta_eps
    elif prediction_type == "v_prediction":
        pred_v = pred
        pred_eps = sigma_t * noisy_latents + alpha_t * pred_v
        delta_eps = (noise - pred_eps).detach()
        delta_eps.mul_(dream_lambda)
        updated_noisy_latents = noisy_latents + sigma_t * delta_eps
        updated_target = target + alpha_t * delta_eps
    else:
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")

    return updated_noisy_latents, updated_target
