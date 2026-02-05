from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch

from ..config import ExperimentConfig
from .batch import TextToImageBatch
from .interfaces import NoiseSchedulerProtocol, TextEncoderProtocol, UNetProtocol, VAEProtocol
from .train_step import TrainStepOutput, train_step

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerState:
    """State tracked by the training runner.

    Attributes:
        global_step: Number of optimizer steps performed.
        epoch: Current epoch index.
        step_in_epoch: Step index within current epoch.
    """

    global_step: int
    epoch: int
    step_in_epoch: int


def _resolve_precision(precision: str, device: torch.device) -> Tuple[str, Optional[torch.dtype], bool]:
    precision_lower = precision.lower()
    supported = {"fp32", "fp16", "bf16"}
    if precision_lower not in supported:
        raise ValueError(f"Unsupported precision '{precision}'. Expected one of {sorted(supported)}.")

    if device.type != "cuda" and precision_lower != "fp32":
        logger.warning("precision=%s requested on non-CUDA device; falling back to fp32", precision_lower)
        precision_lower = "fp32"

    if precision_lower == "bf16" and device.type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            logger.warning("bf16 not supported on this CUDA device; falling back to fp32")
            precision_lower = "fp32"

    if precision_lower == "fp16":
        return precision_lower, torch.float16, True
    if precision_lower == "bf16":
        return precision_lower, torch.bfloat16, False
    return precision_lower, None, False


def _load_checkpoint(
    *,
    checkpoint_path: Path,
    unet: UNetProtocol,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
) -> int:
    payload = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(payload["unet"])
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None:
        scaler_state = payload.get("scaler")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        else:
            logger.warning("checkpoint missing scaler state; continuing without scaler restore")
    elif payload.get("scaler") is not None:
        logger.warning("checkpoint has scaler state but AMP is disabled in current run")
    return int(payload.get("global_step", 0))


def _save_checkpoint(
    *,
    output_dir: Path,
    global_step: int,
    unet: UNetProtocol,
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Path:
    checkpoint_path = output_dir / f"checkpoint-{global_step}.pt"
    payload = {
        "global_step": global_step,
        "unet": unet.state_dict(),
        "optimizer": optimizer.state_dict(),
        "precision": cfg.train.precision,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": asdict(cfg),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def run_text_to_image_training(
    *,
    cfg: ExperimentConfig,
    unet: UNetProtocol,
    vae: VAEProtocol,
    text_encoder: TextEncoderProtocol,
    noise_scheduler: NoiseSchedulerProtocol,
    optimizer: torch.optim.Optimizer,
    train_dataloader: Iterable[TextToImageBatch],
    output_dir: Path,
    device: torch.device,
) -> RunnerState:
    """Run a minimal text-to-image training loop.

    Args:
        cfg: Experiment configuration.
        unet: Conditional UNet model.
        vae: Variational autoencoder for latents.
        text_encoder: Text encoder (e.g., CLIP).
        noise_scheduler: Noise scheduler.
        optimizer: Optimizer for UNet parameters.
        train_dataloader: Iterable of TextToImageBatch.
        output_dir: Directory to save checkpoints.
        device: Target device.

    Returns:
        RunnerState after training completes.
    """
    if cfg.train.grad_accum_steps < 1:
        raise ValueError("train.grad_accum_steps must be >= 1")
    if cfg.train.max_steps < 1:
        raise ValueError("train.max_steps must be >= 1")
    if cfg.train.checkpointing_steps < 1:
        raise ValueError("train.checkpointing_steps must be >= 1")
    if cfg.train.log_every < 1:
        raise ValueError("train.log_every must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)

    precision, amp_dtype, use_scaler = _resolve_precision(cfg.train.precision, device)
    cfg.train.precision = precision
    logger.info("training precision: %s", precision)
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    if use_scaler:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    unet.train()
    if hasattr(vae, "eval"):
        vae.eval()
    if hasattr(text_encoder, "eval"):
        text_encoder.eval()

    steps_per_epoch = math.ceil(len(train_dataloader) / cfg.train.grad_accum_steps)
    num_epochs = math.ceil(cfg.train.max_steps / steps_per_epoch)

    global_step = 0
    if cfg.train.resume_from:
        ckpt_path = Path(cfg.train.resume_from).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume_from checkpoint not found: {ckpt_path}")
        logger.info("resuming from checkpoint: %s", ckpt_path)
        global_step = _load_checkpoint(
            checkpoint_path=ckpt_path,
            unet=unet,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

    running_loss = 0.0
    step_in_epoch = 0

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            step_in_epoch = step
            if global_step >= cfg.train.max_steps:
                break
            if not isinstance(batch, TextToImageBatch):
                raise TypeError("train_dataloader must yield TextToImageBatch")

            batch = TextToImageBatch(
                pixel_values=batch.pixel_values.to(device),
                input_ids=batch.input_ids.to(device),
                metadata=batch.metadata,
            )

            if amp_dtype is not None:
                autocast_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype)
            else:
                autocast_ctx = nullcontext()
            with autocast_ctx:
                step_out: TrainStepOutput = train_step(
                    unet=unet,
                    vae=vae,
                    text_encoder=text_encoder,
                    noise_scheduler=noise_scheduler,
                    batch=batch,
                    prediction_type=cfg.model.prediction_type,
                    dream_training=cfg.train.dream_training,
                    dream_detail_preservation=cfg.train.dream_detail_preservation,
                )

            loss = step_out.loss / cfg.train.grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += step_out.loss.detach().item()

            if (step + 1) % cfg.train.grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg.train.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % cfg.train.log_every == 0:
                    avg_loss = running_loss / float(cfg.train.log_every)
                    logger.info("step=%d loss=%.6f", global_step, avg_loss)
                    running_loss = 0.0

                if global_step % cfg.train.checkpointing_steps == 0:
                    ckpt_path = _save_checkpoint(
                        output_dir=output_dir,
                        global_step=global_step,
                        unet=unet,
                        optimizer=optimizer,
                        cfg=cfg,
                        scaler=scaler,
                    )
                    logger.info("saved checkpoint: %s", ckpt_path)

            if global_step >= cfg.train.max_steps:
                break

    return RunnerState(global_step=global_step, epoch=epoch, step_in_epoch=step_in_epoch)
