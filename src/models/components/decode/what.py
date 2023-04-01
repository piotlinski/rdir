"""DIR decoder for z_what."""
import math

import torch
import torch.nn as nn

from src.models.components import build_conv2d_block


class WhatDecoder(nn.Module):
    """Module decoding latent z_what code to individual images."""

    def __init__(
        self, latent_dim: int = 64, decoded_size: int = 64, channels: int = 64
    ):
        """
        :param latent_dim: latent representation size
        :param decoded_size: reconstructed object image size
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.decoded_size = decoded_size
        self.channels = channels

        power = int(math.log2(self.decoded_size) - 3)  # 3 is for 4x4 decreased by one
        layers = [
            build_conv2d_block(
                in_channels=self.latent_dim,
                out_channels=self.channels * 2**power,
                cls=nn.ConvTranspose2d,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            )
        ]
        for p in range(power, 0, -1):
            layers.append(
                build_conv2d_block(
                    in_channels=self.channels * 2**p,
                    out_channels=self.channels * 2 ** (p - 1),
                    cls=nn.ConvTranspose2d,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
        layers.extend(
            [
                nn.ConvTranspose2d(
                    self.channels, 3, kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.Sigmoid(),
            ]
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, z_what: torch.Tensor) -> torch.Tensor:
        """Takes z_what latent and outputs decoded images."""
        return self.decoder(z_what.view(-1, self.latent_dim, 1, 1))
