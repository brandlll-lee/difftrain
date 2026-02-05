from __future__ import annotations

from pathlib import Path

import pytest

from difftrain.utils.dummy_data import create_dummy_imagefolder

PIL = pytest.importorskip("PIL")


def test_create_dummy_imagefolder(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    paths = create_dummy_imagefolder(output_dir=output_dir, num_images=3, image_size=32, seed=1)

    assert (output_dir / "metadata.jsonl").exists()
    assert len(paths) == 3
    for path in paths:
        assert path.exists()
