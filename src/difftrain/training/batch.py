from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class TextToImageBatch:
    """Standardized batch for text-to-image training.

    Attributes:
        pixel_values: Input images, normalized to [-1, 1].
        input_ids: Token ids for text encoder.
        metadata: Optional extra fields for logging/debugging.
    """

    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    metadata: Optional[Dict[str, Any]] = None
