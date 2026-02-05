from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import numpy as np


def create_dummy_imagefolder(
    *,
    output_dir: Path,
    num_images: int = 4,
    image_size: int = 64,
    seed: int = 42,
) -> List[Path]:
    """Create a tiny imagefolder dataset with metadata.jsonl.

    Args:
        output_dir: Root directory for the dataset.
        num_images: Number of images to generate.
        image_size: Square image size (pixels).
        seed: Random seed for reproducibility.

    Returns:
        List of image file paths created.

    Raises:
        ImportError: If Pillow is not available.
    """
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Pillow is required to create dummy images.") from exc

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    random.seed(seed)

    image_paths: List[Path] = []
    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for idx in range(num_images):
            image_name = f"{idx:05d}.png"
            image_path = output_dir / image_name

            pixels = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
            Image.fromarray(pixels, mode="RGB").save(image_path)
            image_paths.append(image_path)

            record = {"file_name": image_name, "text": f"dummy caption {idx}"}
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    return image_paths
