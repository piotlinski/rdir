"""Latents handler for DIR."""
from typing import Tuple

import torch

DIRLatents = Tuple[
    torch.Tensor,
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]
DIRRepresentation = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
