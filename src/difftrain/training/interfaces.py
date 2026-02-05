from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class NoiseSchedulerProtocol(Protocol):
    """Protocol for noise schedulers used by training.

    Attributes:
        alphas_cumprod: 1D tensor of cumulative alpha values.
        prediction_type: Prediction mode ("epsilon", "v_prediction", "sample").
    """

    alphas_cumprod: torch.Tensor
    prediction_type: str

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to clean latents.

        Args:
            x0: Clean latents.
            noise: Noise tensor.
            timesteps: Timesteps tensor.

        Returns:
            Noisy latents.
        """

    def get_velocity(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute velocity target for v-prediction.

        Args:
            x0: Clean latents.
            noise: Noise tensor.
            timesteps: Timesteps tensor.

        Returns:
            Velocity tensor.
        """


@runtime_checkable
class VAEProtocol(Protocol):
    """Protocol for VAE used to encode images to latents."""

    scaling_factor: float

    def encode(self, pixel_values: torch.Tensor):
        """Encode images into latent distribution or latents.

        Args:
            pixel_values: Input images.

        Returns:
            An object with a latent distribution or the latents tensor.
        """


@runtime_checkable
class TextEncoderProtocol(Protocol):
    """Protocol for text encoder used for conditioning."""

    def __call__(self, input_ids: torch.Tensor, **kwargs):
        """Encode input ids into hidden states.

        Args:
            input_ids: Token ids.
            **kwargs: Optional encoder kwargs.

        Returns:
            Encoder outputs or hidden states.
        """


@runtime_checkable
class UNetProtocol(Protocol):
    """Protocol for conditional UNet models."""

    def __call__(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ):
        """Predict noise or velocity.

        Args:
            noisy_latents: Noisy latents.
            timesteps: Timesteps tensor.
            encoder_hidden_states: Conditioning states.
            **kwargs: Optional model kwargs.

        Returns:
            Model outputs (Tensor or object with .sample).
        """
