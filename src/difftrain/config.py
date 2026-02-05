from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class TrainConfig:
    seed: int = 42
    output_dir: str = "runs/exp"
    resume_from: str = ""
    precision: str = "fp32"  # fp32|fp16|bf16
    grad_accum_steps: int = 1
    max_steps: int = 1000
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 500
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    checkpointing_steps: int = 500
    report_to: str = "tensorboard"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    dream_training: bool = False
    dream_detail_preservation: float = 1.0


@dataclass
class DataConfig:
    dataset_type: str = "imagefolder"
    data_root: str = "data"
    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 4
    image_column: str = "image"
    caption_column: str = "text"
    center_crop: bool = False
    random_flip: bool = False
    image_interpolation: str = "lanczos"


@dataclass
class ModelConfig:
    prediction_type: str = "epsilon"  # epsilon|v_prediction|sample
    model_name: str = "unet_base"
    pretrained_model_name_or_path: str = ""
    revision: str = ""
    variant: str = ""


@dataclass
class SchedulerConfig:
    name: str = "ddpm"
    num_train_timesteps: int = 1000
    beta_schedule: str = "linear"


@dataclass
class LossConfig:
    loss_type: str = "mse"  # mse|huber
    p2_gamma: float = 0.0
    p2_k: float = 1.0
    snr_weighting: bool = False


@dataclass
class ExperimentConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    return ExperimentConfig(
        train=TrainConfig(**raw.get("train", {})),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        scheduler=SchedulerConfig(**raw.get("scheduler", {})),
        loss=LossConfig(**raw.get("loss", {})),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    payload = {
        "train": asdict(config.train),
        "data": asdict(config.data),
        "model": asdict(config.model),
        "scheduler": asdict(config.scheduler),
        "loss": asdict(config.loss),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

