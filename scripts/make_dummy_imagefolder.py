from __future__ import annotations

import argparse
from pathlib import Path

from difftrain.utils.dummy_data import create_dummy_imagefolder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny imagefolder dataset for smoke tests.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for the dataset.")
    parser.add_argument("--num-images", type=int, default=4, help="Number of images to generate.")
    parser.add_argument("--image-size", type=int, default=64, help="Image size (square).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_dummy_imagefolder(
        output_dir=Path(args.output_dir),
        num_images=args.num_images,
        image_size=args.image_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
