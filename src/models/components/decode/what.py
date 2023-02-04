"""DIR decoder for z_what."""
import math

import torch
import torch.nn as nn


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
            nn.ConvTranspose2d(
                self.latent_dim,
                self.channels * 2**power,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            )
        ]
        for p in range(power, 0, -1):
            segment = [
                nn.BatchNorm2d(self.channels * 2**p),
                nn.LeakyReLU(True),
                nn.ConvTranspose2d(
                    self.channels * 2**p,
                    self.channels * 2 ** (p - 1),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            ]
            layers.extend(segment)
        final_segment = [
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(
                self.channels,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Sigmoid(),
        ]
        layers.extend(final_segment)

        self.decoder = nn.Sequential(*layers)

    def forward(self, z_what: torch.Tensor) -> torch.Tensor:
        """Takes z_what latent and outputs decoded images."""
        return self.decoder(z_what.view(-1, self.latent_dim, 1, 1))
