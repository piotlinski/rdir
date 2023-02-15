"""DIR encoder for z_what."""
from typing import Dict, Tuple

import torch
import torch.nn as nn


class WhatEncoder(nn.Module):
    """Module for encoding input image features to what latent representation."""

    def __init__(
        self,
        latent_dim: int,
        num_hidden: int,
        anchors: Dict[int, int],
        out_channels: Dict[int, int],
        scale_const: float = -1.0,
    ):
        """
        :param latent_dim: latent representation size
        :param num_hidden: number of hidden layers in each encoder
        :param anchors: dictionary containing number of anchors per latents level
        :param out_channels: dictionary containing layer id and number of channels
        :param scale_const: allows to set z_what scale to a constant
            (negative for trainable, 0 for deterministic z_what)
        """
        super().__init__()
        self.is_probabilistic = True

        self.latent_dim = latent_dim
        self.num_hidden = num_hidden
        self.anchors = anchors
        self.out_channels = out_channels

        self.loc_encoders = self._build_encoders()

        self.scale_const = scale_const
        self.scale_encoders = None
        if self.scale_const < 0:
            self.scale_encoders = self._build_encoders()
        if self.scale_const == 0:
            self.is_probabilistic = False

    def _build_feature_encoder(self, in_channels: int, num_anchors: int) -> nn.Module:
        """Prepare single feature encoder."""
        out_size = num_anchors * self.latent_dim
        hidden_size = 2 * out_size

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(True),
        ]
        for _ in range(self.num_hidden):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels=hidden_size,
                        out_channels=hidden_size,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_size),
                    nn.LeakyReLU(True),
                ]
            )
        layers.append(
            nn.Conv2d(in_channels=hidden_size, out_channels=out_size, kernel_size=1)
        )
        return nn.Sequential(*layers)

    def _build_encoders(self) -> nn.ModuleDict:
        """Build models for encoding latent representation."""
        layers = {}
        for idx, num_anchors in self.anchors.items():
            in_channels = self.out_channels[idx]
            layers[str(idx)] = self._build_feature_encoder(
                in_channels=in_channels, num_anchors=num_anchors
            )
        return nn.ModuleDict(layers)

    def forward(self, x: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict z_what loc and scale."""
        locs = []
        scales = []
        for idx, feature in x.items():
            batch_size = feature.shape[0]
            loc = self.loc_encoders[str(idx)](feature)
            loc = (
                loc.permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, self.latent_dim)
            )
            locs.append(loc)

            if self.scale_encoders is not None:
                scale = torch.exp(self.scale_encoders[str(idx)](feature))
                scale = (
                    scale.permute(0, 2, 3, 1)
                    .contiguous()
                    .view(batch_size, -1, self.latent_dim)
                )
                scales.append(scale)
            elif self.is_probabilistic:
                scales.append(
                    loc.new_full((1,), fill_value=self.scale_const).expand_as(loc)
                )

        z_what_loc = torch.cat(locs, dim=1)
        z_what_scale = torch.cat(scales, dim=1) if scales else torch.empty(0)

        return z_what_loc, z_what_scale
