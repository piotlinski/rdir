"""DIR encoder for z_depth."""
from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.components import build_conv2d_block


class DepthEncoder(nn.Module):
    """Module for encoding input image features to depth latent representation."""

    def __init__(
        self,
        anchors: Dict[int, int],
        out_channels: Dict[int, int],
        scale_const: float = -1.0,
    ):
        """
        :param anchors: dictionary containing number of anchors per latents level
        :param out_channels:dictionary containing layer id and number of channels
        :param scale_const: allows to set z_depth scale to a constant
            (negative for trainable, 0 for deterministic z_depth)
        """
        super().__init__()
        self.is_probabilistic = True

        self.anchors = anchors
        self.out_channels = out_channels

        self.loc_encoders = self._build_encoders()

        self.scale_const = scale_const
        self.scale_encoders = None
        if self.scale_const < 0:
            self.scale_encoders = self._build_encoders()
        if self.scale_const == 0:
            self.is_probabilistic = False

    def _build_encoders(self) -> nn.ModuleDict:
        """Build models for encoding latent representation."""
        layers = {}
        for idx, num_anchors in self.anchors.items():
            in_channels = self.out_channels[idx]
            hidden_dim = in_channels // 2
            layers[str(idx)] = nn.Sequential(
                build_conv2d_block(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.Conv2d(
                    in_channels=hidden_dim, out_channels=num_anchors, kernel_size=1
                ),
            )
        return nn.ModuleDict(layers)

    def forward(self, x: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict z_depth loc and scale."""
        locs = []
        scales = []
        for idx, feature in x.items():
            batch_size = feature.shape[0]
            loc = self.loc_encoders[str(idx)](feature)
            loc = loc.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
            locs.append(loc)

            if self.scale_encoders is not None:
                scale = torch.exp(self.scale_encoders[str(idx)](feature))
                scale = scale.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
                scales.append(scale)
            elif self.is_probabilistic:
                scales.append(
                    loc.new_full((1,), fill_value=self.scale_const).expand_as(loc)
                )

        z_depth_loc = torch.cat(locs, dim=1)
        z_depth_scale = torch.cat(scales, dim=1) if scales else torch.empty(0)

        return z_depth_loc, z_depth_scale
