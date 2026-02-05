from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .batch import TextToImageBatch
from .dream import compute_dream_and_update_latents
from .interfaces import NoiseSchedulerProtocol, TextEncoderProtocol, UNetProtocol, VAEProtocol


@dataclass(frozen=True)
class TrainStepOutput:
    """Outputs from a single training step.

    Attributes:
        loss: Scalar loss tensor.
        model_pred: Raw model prediction.
        target: Target tensor used to compute loss.
        timesteps: Timesteps sampled for this batch.
        noisy_latents: Noisy latents fed into the model.
        latents: Clean latents produced by the VAE.
    """

    loss: torch.Tensor
    model_pred: torch.Tensor
    target: torch.Tensor
    timesteps: torch.Tensor
    noisy_latents: torch.Tensor
    latents: torch.Tensor


def _get_prediction_type(noise_scheduler: NoiseSchedulerProtocol, override: Optional[str]) -> str:
    if override is not None:
        _set_prediction_type(noise_scheduler, override)
        return override
    if hasattr(noise_scheduler, "config") and hasattr(noise_scheduler.config, "prediction_type"):
        return str(noise_scheduler.config.prediction_type)
    return str(noise_scheduler.prediction_type)


def _set_prediction_type(noise_scheduler: NoiseSchedulerProtocol, prediction_type: str) -> None:
    if hasattr(noise_scheduler, "register_to_config"):
        noise_scheduler.register_to_config(prediction_type=prediction_type)
        return
    if hasattr(noise_scheduler, "config") and hasattr(noise_scheduler.config, "prediction_type"):
        noise_scheduler.config.prediction_type = prediction_type
        return
    if hasattr(noise_scheduler, "prediction_type"):
        noise_scheduler.prediction_type = prediction_type


def _get_num_train_timesteps(noise_scheduler: NoiseSchedulerProtocol) -> int:
    if hasattr(noise_scheduler, "config") and hasattr(noise_scheduler.config, "num_train_timesteps"):
        return int(noise_scheduler.config.num_train_timesteps)
    if hasattr(noise_scheduler, "num_train_timesteps"):
        return int(noise_scheduler.num_train_timesteps)
    raise AttributeError("noise_scheduler must expose num_train_timesteps or config.num_train_timesteps")


def _unwrap_prediction(pred: object) -> torch.Tensor:
    if isinstance(pred, torch.Tensor):
        return pred
    if isinstance(pred, tuple) and pred:
        return pred[0]
    if hasattr(pred, "sample"):
        return pred.sample
    raise TypeError("UNet output must be a Tensor or have a .sample attribute")


def _unwrap_text_encoder_output(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output:
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError("Text encoder output must be a Tensor, tuple, or have .last_hidden_state")


def _encode_latents(vae: VAEProtocol, pixel_values: torch.Tensor) -> torch.Tensor:
    enc = vae.encode(pixel_values)
    if hasattr(enc, "latent_dist"):
        latents = enc.latent_dist.sample()
    elif hasattr(enc, "sample"):
        latents = enc.sample()
    elif isinstance(enc, torch.Tensor):
        latents = enc
    else:
        raise TypeError("VAE encode output must be Tensor or have latent_dist/sample")
    scaling_factor = getattr(vae, "scaling_factor", None)
    if scaling_factor is None and hasattr(vae, "config"):
        scaling_factor = getattr(vae.config, "scaling_factor", None)
    if scaling_factor is None:
        raise AttributeError("VAE must expose scaling_factor or config.scaling_factor")
    return latents * float(scaling_factor)


def train_step(
    *,
    unet: UNetProtocol,
    vae: VAEProtocol,
    text_encoder: TextEncoderProtocol,
    noise_scheduler: NoiseSchedulerProtocol,
    batch: TextToImageBatch,
    prediction_type: Optional[str] = None,
    dream_training: bool = False,
    dream_detail_preservation: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> TrainStepOutput:
    """Run a single training step for text-to-image diffusion.

    Args:
        unet: Conditional UNet model.
        vae: Variational autoencoder.
        text_encoder: Text encoder (e.g., CLIPTextModel).
        noise_scheduler: Noise scheduler.
        batch: Standardized text-to-image batch.
        prediction_type: Override prediction type ("epsilon", "v_prediction", "sample").
        dream_training: Whether to apply DREAM update.
        dream_detail_preservation: DREAM detail preservation factor p.
        generator: Optional random generator for deterministic sampling.

    Returns:
        TrainStepOutput with loss and intermediate tensors.

    Raises:
        ValueError: If prediction_type is unsupported.
    """
    latents = _encode_latents(vae, batch.pixel_values)
    bsz = latents.shape[0]

    num_train_timesteps = _get_num_train_timesteps(noise_scheduler)
    timesteps = torch.randint(
        0,
        num_train_timesteps,
        (bsz,),
        device=latents.device,
        dtype=torch.long,
        generator=generator,
    )

    noise = torch.randn(
        latents.shape,
        device=latents.device,
        dtype=latents.dtype,
        generator=generator,
    )
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    encoder_out = text_encoder(batch.input_ids)
    encoder_hidden_states = _unwrap_text_encoder_output(encoder_out)

    pred_type = _get_prediction_type(noise_scheduler, prediction_type)
    if pred_type == "epsilon":
        target = noise
    elif pred_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    elif pred_type == "sample":
        target = noisy_latents
    else:
        raise ValueError(f"Unsupported prediction_type: {pred_type}")

    if dream_training:
        noisy_latents, target = compute_dream_and_update_latents(
            unet=unet,
            noise_scheduler=noise_scheduler,
            timesteps=timesteps,
            noise=noise,
            noisy_latents=noisy_latents,
            target=target,
            encoder_hidden_states=encoder_hidden_states,
            dream_detail_preservation=dream_detail_preservation,
        )

    model_pred = _unwrap_prediction(unet(noisy_latents, timesteps, encoder_hidden_states))
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    return TrainStepOutput(
        loss=loss,
        model_pred=model_pred,
        target=target,
        timesteps=timesteps,
        noisy_latents=noisy_latents,
        latents=latents,
    )
