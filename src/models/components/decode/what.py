"""DIR decoder for z_what."""
import math

import torch
import torch.nn as nn


class WhatDecoder(nn.Module):
    """Module decoding latent z_what code to individual images."""

    def __init__(self, latent_dim: int = 64, decoded_size: int = 64):
        """
        :param latent_dim: latent representation size
        :param decoded_size: reconstructed object image size
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.decoded_size = decoded_size

        channels = 8
        layers = [
            nn.Sigmoid(),
            nn.Conv2d(channels, 3, kernel_size=1),
        ]
        for _ in range(int(math.log2(self.decoded_size))):
            layers.append(nn.LeakyReLU())
            layers.append(
                nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)
            )
            channels *= 2
        layers.append(nn.Conv2d(self.latent_dim, channels, kernel_size=1))

        self.decoder = nn.Sequential(*layers[::-1])

    def forward(self, z_what: torch.Tensor) -> torch.Tensor:
        """Takes z_what latent and outputs decoded images."""
        return self.decoder(z_what.view(-1, self.latent_dim, 1, 1))
