from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest
import torch
from torch.utils.data import DataLoader

from difftrain.config import ExperimentConfig, load_config, save_config
from difftrain.training import TextToImageBatch, run_text_to_image_training
from difftrain.utils.dummy_data import create_dummy_imagefolder
from difftrain.utils.manifest import write_run_manifest
from difftrain.utils.repro import resolve_output_dir, set_seed, write_env_report

from .testing_utils import require_text2image_deps


pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.cuda, pytest.mark.text2image]


def _build_dataloader(cfg: ExperimentConfig, tokenizer) -> DataLoader:
    import datasets
    from torchvision import transforms

    if cfg.data.dataset_type != "imagefolder":
        raise ValueError("Only dataset_type=imagefolder is supported in this smoke test.")

    data_root = Path(cfg.data.data_root).expanduser()
    dataset = datasets.load_dataset("imagefolder", data_files={"train": str(data_root / "**")})
    column_names = dataset["train"].column_names
    if cfg.data.image_column not in column_names:
        raise ValueError(f"image_column '{cfg.data.image_column}' not in dataset columns: {column_names}")
    if cfg.data.caption_column not in column_names:
        raise ValueError(f"caption_column '{cfg.data.caption_column}' not in dataset columns: {column_names}")

    def tokenize_captions(examples) -> List[torch.Tensor]:
        captions = []
        for caption in examples[cfg.data.caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, tuple)):
                captions.append(caption[0])
            else:
                raise ValueError("Caption must be a string or list of strings.")
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    interpolation = getattr(transforms.InterpolationMode, cfg.data.image_interpolation.upper(), None)
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation mode: {cfg.data.image_interpolation}")

    train_transforms = transforms.Compose(
        [
            transforms.Resize(cfg.data.image_size, interpolation=interpolation),
            (
                transforms.CenterCrop(cfg.data.image_size)
                if cfg.data.center_crop
                else transforms.RandomCrop(cfg.data.image_size)
            ),
            transforms.RandomHorizontalFlip() if cfg.data.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[cfg.data.image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples]).float()
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        return TextToImageBatch(pixel_values=pixel_values, input_ids=input_ids)

    return DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )


def _load_models(cfg: ExperimentConfig):
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    model_id = cfg.model.pretrained_model_name_or_path
    if not model_id:
        raise ValueError("model.pretrained_model_name_or_path must be set.")

    revision = cfg.model.revision or None
    variant = cfg.model.variant or None

    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", revision=revision)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", revision=revision, variant=variant)
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        revision=revision,
        variant=variant,
    )

    return noise_scheduler, tokenizer, text_encoder, vae, unet


def _write_repro_artifacts(cfg: ExperimentConfig, output_dir: Path, config_path: Path) -> None:
    env_path = write_env_report(output_dir / "env.json")
    resolved_cfg_path = output_dir / "config.resolved.yaml"
    save_config(cfg, resolved_cfg_path)
    write_run_manifest(
        output_dir=output_dir,
        config_path=config_path,
        cfg=cfg,
        deterministic=True,
        env_json_path=env_path,
        resolved_config_path=resolved_cfg_path,
        repo_dir=Path.cwd(),
    )


def _offline_mode_enabled() -> bool:
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        value = os.environ.get(key, "")
        if value.lower() in ("1", "true", "yes", "on"):
            return True
    return False


def test_text2image_train_gpu_smoke(tmp_path: Path) -> None:
    require_text2image_deps()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this smoke test.")
    if _offline_mode_enabled():
        pytest.skip("Offline mode enabled; cannot download model weights.")

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "smoke_text2image.yaml"
    cfg = load_config(config_path)

    cfg.train.output_dir = str(tmp_path / "run")
    cfg.data.data_root = str(tmp_path / "data")
    cfg.data.batch_size = 1
    cfg.data.num_workers = 0
    cfg.train.max_steps = min(int(cfg.train.max_steps), 2)
    cfg.train.checkpointing_steps = 1
    cfg.train.log_every = 1

    dataset_dir = Path(cfg.data.data_root)
    create_dummy_imagefolder(
        output_dir=dataset_dir,
        num_images=4,
        image_size=cfg.data.image_size,
        seed=cfg.train.seed,
    )

    set_seed(cfg.train.seed, deterministic=True)
    output_dir = resolve_output_dir(cfg.train.output_dir)
    _write_repro_artifacts(cfg, output_dir, config_path)

    noise_scheduler, tokenizer, text_encoder, vae, unet = _load_models(cfg)
    train_dataloader = _build_dataloader(cfg=cfg, tokenizer=tokenizer)
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=cfg.train.learning_rate,
        betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
        weight_decay=cfg.train.adam_weight_decay,
        eps=cfg.train.adam_epsilon,
    )

    device = torch.device("cuda")
    unet.to(device)
    vae.to(device)
    text_encoder.to(device)

    if hasattr(unet, "enable_gradient_checkpointing") and cfg.train.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if hasattr(vae, "requires_grad_"):
        vae.requires_grad_(False)
    if hasattr(text_encoder, "requires_grad_"):
        text_encoder.requires_grad_(False)

    state = run_text_to_image_training(
        cfg=cfg,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        output_dir=output_dir,
        device=device,
    )

    assert state.global_step >= cfg.train.max_steps
    assert (output_dir / "env.json").exists()
    assert (output_dir / "config.resolved.yaml").exists()
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "checkpoint-1.pt").exists()
    if cfg.train.max_steps >= 2:
        assert (output_dir / "checkpoint-2.pt").exists()
