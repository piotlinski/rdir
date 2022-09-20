"""Latents handler for DIR."""
from typing import Tuple

import torch
from torch import nn

DIRLatents = Tuple[
    torch.Tensor,
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]
DIRRepresentation = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class LatentHandler(nn.Module):
    """Module for handling latents."""

    def __init__(
        self,
        reset_non_present: bool = True,
    ):
        """
        :param reset_non_present: set non-present latents to some ordinary ones
        """
        super().__init__()

        self._reset_non_present = reset_non_present

        self._z_present_eps = 1e-3
        self.register_buffer("_empty_loc", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer("_empty_scale", torch.tensor(1.0, dtype=torch.float))

    @staticmethod
    def reset_latent(
        latent: torch.Tensor, mask: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Reset latents not covered by mask to given value."""
        return torch.where(mask, latent, value.type(latent.dtype))

    def reset_non_present(self, latents: DIRLatents) -> DIRLatents:
        """Reset latents, whose z_present is 0."""
        (
            z_where,
            z_present,
            (z_what_loc, z_what_scale),
            (z_depth_loc, z_depth_scale),
        ) = latents
        mask = torch.gt(z_present, self._z_present_eps)
        z_what_loc = self.reset_latent(z_what_loc, mask, self._empty_loc)
        z_what_scale = self.reset_latent(z_what_scale, mask, self._empty_scale)
        z_depth_loc = self.reset_latent(z_depth_loc, mask, self._empty_loc)
        z_depth_scale = self.reset_latent(z_depth_scale, mask, self._empty_scale)
        return (
            z_where,
            z_present,
            (z_what_loc, z_what_scale),
            (z_depth_loc, z_depth_scale),
        )

    def forward(self, latents: DIRLatents) -> DIRLatents:
        if self._reset_non_present:
            latents = self.reset_non_present(latents)

        return latents
