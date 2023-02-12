"""DIR decoder."""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.decode.what import WhatDecoder
from src.models.components.decode.where import WhereTransformer
from src.models.components.latents import DIRRepresentation


class Decoder(nn.Module):
    """Module decoding latent representation."""

    def __init__(
        self,
        z_what_size: int = 64,
        decoded_size: int = 64,
        decoder_channels: int = 64,
        square_boxes: bool = False,
        image_size: int = 416,
        train_what: bool = True,
        include_negative: bool = False,
    ):
        """
        :param z_what_size: z_what latent representation size
        :param decoded_size: reconstructed object size
        :param decoder_channels: number of channels in decoder
        :param square_boxes: use square bounding boxes
        :param image_size: reconstructed image size
        :param train_what: perform z_what decoder training
        :param include_negative: include negative objects in reconstruction
        """
        super().__init__()

        self.image_size = image_size
        self.include_negative = include_negative

        self.what_dec = WhatDecoder(
            latent_dim=z_what_size, decoded_size=decoded_size, channels=decoder_channels
        ).requires_grad_(train_what)
        self.where_stn = WhereTransformer(image_size=image_size, square=square_boxes)

        self.register_buffer("_no_object", torch.tensor([0], dtype=torch.float))

    def decode_objects(self, z_what: torch.Tensor) -> torch.Tensor:
        """Decode z_what to acquire individual objects and their z_where location."""
        z_what_flat = z_what.view(-1, z_what.shape[-1])
        decoded_objects = self.what_dec(z_what_flat)
        return decoded_objects

    def transform_objects(
        self, decoded_objects: torch.Tensor, z_where: torch.Tensor
    ) -> torch.Tensor:
        """Render reconstructions and their depths."""
        batch_size, n_objects, *_ = z_where.shape
        z_where_flat = z_where.view(-1, z_where.shape[-1])
        objects = self.where_stn(decoded_objects, z_where_flat)
        return objects.view(batch_size, n_objects, *objects.shape[-3:])

    def filter(self, objects: torch.Tensor, z_present: torch.Tensor) -> torch.Tensor:
        """Set non-present objects to zeros."""
        batch_size, n_objects, *_ = z_present.shape
        mask = (z_present == -1).view(batch_size, n_objects, 1, 1, 1).expand_as(objects)
        no_objects = self._no_object.view(1, 1, 1, 1, 1).expand_as(objects)
        return torch.where(mask, no_objects, objects)

    @staticmethod
    def reconstruct(objects: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Combine decoder images into one by weighted sum."""
        weighted_objects = objects * F.softmax(weights, dim=1).view(
            *weights.shape[:2], 1, 1, 1
        )
        merged = torch.sum(weighted_objects, dim=1)
        return merged

    @staticmethod
    def normalize_reconstructions(reconstructions: torch.Tensor) -> torch.Tensor:
        """Normalize reconstructions to fit range 0-1."""
        batch_size = reconstructions.shape[0]
        max_values, _ = reconstructions.view(batch_size, -1).max(dim=1, keepdim=True)
        max_values = (
            max_values.unsqueeze(-1).unsqueeze(-1).expand_as(reconstructions) + 1e-3
        )
        return reconstructions / max_values

    def forward(
        self,
        representation: DIRRepresentation,
        return_objects: bool = False,
        normalize_reconstructions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct images from representation."""
        ret = {}

        z_where, z_present, z_what, z_depth = representation

        objects = self.decode_objects(z_what)

        if return_objects:
            ret["objects"] = objects

        objects = self.transform_objects(objects, z_where)

        if not self.training or not self.include_negative:
            objects = self.filter(objects, z_present)

        reconstructions = self.reconstruct(objects, z_depth)

        if normalize_reconstructions:
            reconstructions = self.normalize_reconstructions(reconstructions)

        ret["reconstructions"] = reconstructions

        return ret
