from __future__ import annotations

from typing import Type

import pytest
import torch

from difftrain.training import TextToImageBatch, train_step

from .fixtures import DummyScheduler


def _make_batch() -> TextToImageBatch:
    pixel_values = torch.randn(2, 3, 4, 4)
    input_ids = torch.randint(0, 10, (2, 4))
    return TextToImageBatch(pixel_values=pixel_values, input_ids=input_ids)


def _make_scheduler() -> DummyScheduler:
    alphas_cumprod = torch.tensor([0.9, 0.6, 0.3], dtype=torch.float32)
    return DummyScheduler(alphas_cumprod, prediction_type="epsilon")


class TextEncoderTensor:
    def __call__(self, input_ids: torch.Tensor):
        return input_ids.float().unsqueeze(-1)


class TextEncoderTuple:
    def __call__(self, input_ids: torch.Tensor):
        return (input_ids.float().unsqueeze(-1),)


class _HiddenStateOut:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.last_hidden_state = hidden


class TextEncoderHidden:
    def __call__(self, input_ids: torch.Tensor):
        hidden = input_ids.float().unsqueeze(-1)
        return _HiddenStateOut(hidden)


class UNetTensor(torch.nn.Module):
    def forward(self, noisy_latents, timesteps, encoder_hidden_states):  # noqa: D401
        return noisy_latents


class UNetTuple(torch.nn.Module):
    def forward(self, noisy_latents, timesteps, encoder_hidden_states):  # noqa: D401
        return (noisy_latents,)


class _SampleOut:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class UNetSample(torch.nn.Module):
    def forward(self, noisy_latents, timesteps, encoder_hidden_states):  # noqa: D401
        return _SampleOut(noisy_latents)


class VAEEncodeTensor:
    scaling_factor = 0.5

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values * 0.25


class _SampleOnly:
    def __init__(self, sample: torch.Tensor) -> None:
        self._sample = sample

    def sample(self) -> torch.Tensor:
        return self._sample


class VAEEncodeSample:
    scaling_factor = 0.5

    def encode(self, pixel_values: torch.Tensor) -> _SampleOnly:
        return _SampleOnly(pixel_values * 0.25)


class _LatentDist:
    def __init__(self, sample: torch.Tensor) -> None:
        self._sample = sample

    def sample(self) -> torch.Tensor:
        return self._sample


class _LatentDistOut:
    def __init__(self, sample: torch.Tensor) -> None:
        self.latent_dist = _LatentDist(sample)


class VAEEncodeLatentDist:
    scaling_factor = 0.5

    def encode(self, pixel_values: torch.Tensor) -> _LatentDistOut:
        return _LatentDistOut(pixel_values * 0.25)


@pytest.mark.parametrize("vae_factory", [VAEEncodeTensor, VAEEncodeSample, VAEEncodeLatentDist])
@pytest.mark.parametrize("text_encoder_factory", [TextEncoderTensor, TextEncoderTuple, TextEncoderHidden])
@pytest.mark.parametrize("unet_factory", [UNetTensor, UNetTuple, UNetSample])
def test_train_step_accepts_output_variants(
    vae_factory: Type[object],
    text_encoder_factory: Type[object],
    unet_factory: Type[torch.nn.Module],
) -> None:
    torch.manual_seed(0)
    scheduler = _make_scheduler()
    batch = _make_batch()

    vae = vae_factory()
    text_encoder = text_encoder_factory()
    unet = unet_factory()

    output = train_step(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        noise_scheduler=scheduler,
        batch=batch,
        prediction_type="epsilon",
        dream_training=False,
    )

    assert output.model_pred.shape == output.target.shape
    assert torch.isfinite(output.loss).all()


class VAEWithoutScaling:
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values * 0.25


def test_train_step_raises_when_scaling_factor_missing() -> None:
    scheduler = _make_scheduler()
    batch = _make_batch()

    with pytest.raises(AttributeError, match="scaling_factor"):
        train_step(
            unet=UNetTensor(),
            vae=VAEWithoutScaling(),
            text_encoder=TextEncoderTensor(),
            noise_scheduler=scheduler,
            batch=batch,
            prediction_type="epsilon",
            dream_training=False,
        )


class SchedulerWithoutTimesteps:
    prediction_type = "epsilon"

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return x0 + noise * 0.0


def test_train_step_raises_when_num_train_timesteps_missing() -> None:
    batch = _make_batch()
    scheduler = SchedulerWithoutTimesteps()

    with pytest.raises(AttributeError, match="num_train_timesteps"):
        train_step(
            unet=UNetTensor(),
            vae=VAEEncodeTensor(),
            text_encoder=TextEncoderTensor(),
            noise_scheduler=scheduler,
            batch=batch,
            prediction_type="epsilon",
            dream_training=False,
        )


class SchedulerUnknownPrediction(DummyScheduler):
    def __init__(self) -> None:
        alphas = torch.tensor([0.9, 0.6, 0.3], dtype=torch.float32)
        super().__init__(alphas, prediction_type="unknown")


def test_train_step_raises_on_unknown_prediction_type() -> None:
    batch = _make_batch()
    scheduler = SchedulerUnknownPrediction()

    with pytest.raises(ValueError, match="Unsupported prediction_type"):
        train_step(
            unet=UNetTensor(),
            vae=VAEEncodeTensor(),
            text_encoder=TextEncoderTensor(),
            noise_scheduler=scheduler,
            batch=batch,
            prediction_type=None,
            dream_training=False,
        )
