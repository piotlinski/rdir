"""DIR heads for z_where and z_present."""
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

    def forward(self, confs: torch.Tensor) -> torch.Tensor:
        """Perform forward pass."""
        max_values, _ = torch.max(confs, axis=-1, keepdim=True)
        return max_values
