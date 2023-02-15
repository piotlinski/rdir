"""DIR heads for z_where and z_present."""
from typing import Optional, Tuple

import torch
import torch.nn as nn


class WhereHead(nn.Module):
    """Parser for head output to produce bounding boxes."""

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """Perform forward pass."""
        batch_size, n_boxes, *_ = boxes.shape
        return boxes.view(batch_size, n_boxes, 4)


class PresentHead(nn.Module):
    """Parser for head output to produce object presence."""

    def __init__(self, classes: Optional[Tuple[int, ...]] = None):
        """
        :param classes: list of classes to filter
        """
        super().__init__()
        self.index = slice(None)
        if classes is not None:
            self.index = classes

    def forward(self, confs: torch.Tensor) -> torch.Tensor:
        """Perform forward pass."""
        confs = confs[..., self.index]
        max_values, _ = torch.max(confs, axis=-1, keepdim=True)
        return max_values
