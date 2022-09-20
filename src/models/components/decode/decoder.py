"""DIR decoder."""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.decode.what import WhatDecoder
from src.models.components.decode.where import WhereTransformer
from src.models.components.latents import DIRRepresentation


class Decoder(nn.Module):
    """Module decoding latent representation."""

    EMPTY_DEPTH = -1e3

    def __init__(
        self,
        z_what_size: int = 64,
        decoded_size: int = 64,
        image_size: int = 416,
        train_what: bool = True,
    ):
        """
        :param z_what_size: z_what latent representation size
        :param decoded_size: reconstructed object size
        :param image_size: reconstructed image size
        :param train_what: perform z_what decoder training
        """
        super().__init__()

        self.image_size = image_size

        self.what_dec = WhatDecoder(
            latent_dim=z_what_size, decoded_size=decoded_size
        ).requires_grad_(train_what)
        self.where_stn = WhereTransformer(image_size=image_size)

    @staticmethod
    def fix_z_present(z_present: torch.Tensor) -> torch.Tensor:
        """Review if any image yielded no detection and choose random objects."""
        n_present = torch.sum(z_present, dim=1, dtype=torch.long)
        max_present = torch.max(n_present)

        empty = n_present == 0
        if empty.any():
            indices = torch.argsort(torch.rand(*z_present.shape[:-1]), dim=-1)[
                :, :max_present
            ]
            corrected = torch.zeros_like(z_present)
            corrected[torch.arange(corrected.shape[0]).unsqueeze(-1), indices] = 1
            z_present = torch.where(empty.unsqueeze(-1), corrected, z_present)

        return z_present

    @staticmethod
    def filter(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Filter representation based on mask."""
        return tensor[mask.expand_as(tensor)].view(-1, tensor.shape[-1])

    def filter_representation(
        self, representation: DIRRepresentation
    ) -> DIRRepresentation:
        """Filter representation according to z_present."""
        z_where, z_present, z_what, z_depth = representation
        z_present = self.fix_z_present(z_present)
        present_mask = torch.eq(z_present, 1)
        z_where = self.filter(z_where, present_mask)
        z_what = self.filter(z_what, present_mask)
        z_depth = self.filter(z_depth, present_mask)
        return z_where, z_present, z_what, z_depth

    def decode_objects(
        self, z_where: torch.Tensor, z_what: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode z_what to acquire individual objects and their z_where location."""
        z_where_flat = z_where.view(-1, z_where.shape[-1])
        z_what_flat = z_what.view(-1, z_what.shape[-1])
        decoded_objects = self.what_dec(z_what_flat)
        return decoded_objects, z_where_flat

    @staticmethod
    def pad_indices(n_present: torch.Tensor) -> torch.Tensor:
        """Using number of objects in chunks create indices.

        .. so that every chunk is padded to the same dimension.

        .. Assumes index 0 refers to "starter" (empty) object

        :param n_present: number of objects in each chunk
        :return: indices for padding tensors
        """
        end_idx = 1
        max_objects = torch.max(n_present)
        indices = []
        for chunk_objects in n_present:
            start_idx = end_idx
            end_idx = end_idx + chunk_objects
            idx_range = torch.arange(
                start=start_idx, end=end_idx, dtype=torch.long, device=n_present.device
            )
            start_pad = 1
            end_pad = max_objects - chunk_objects
            indices.append(F.pad(idx_range, pad=[start_pad, end_pad]))
        return torch.cat(indices)

    def pad_reconstructions(
        self,
        transformed_objects: torch.Tensor,
        z_depth: torch.Tensor,
        n_present: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad tensors to have identical 1. dim shape.

        .. and reshape to (batch_size x n_objects x ...)
        """
        image_size = self.image_size

        objects_starter = transformed_objects.new_zeros((1, 3, image_size, image_size))
        z_depth_starter = z_depth.new_full((1, 1), fill_value=self.EMPTY_DEPTH)

        objects = torch.cat((objects_starter, transformed_objects), dim=0)
        z_depth = torch.cat(
            (z_depth_starter, z_depth.view(-1, z_depth.shape[-1])), dim=0
        )

        max_present = torch.max(n_present)
        indices = self.pad_indices(n_present)

        padded_shape = max_present.item() + 1
        objects = objects[indices].view(-1, padded_shape, 3, image_size, image_size)
        z_depth = z_depth[indices].view(-1, padded_shape, 1)

        return objects, z_depth

    def transform_objects(
        self,
        decoded_objects: torch.Tensor,
        z_where_flat: torch.Tensor,
        z_present: torch.Tensor,
        z_depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render reconstructions and their depths."""
        n_present = torch.sum(z_present, dim=1, dtype=torch.long).squeeze(-1)
        objects = self.where_stn(decoded_objects, z_where_flat)
        objects, depths = self.pad_reconstructions(
            transformed_objects=objects, z_depth=z_depth, n_present=n_present
        )
        return objects, depths

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

        z_where, z_present, z_what, z_depth = self.filter_representation(representation)

        objects, z_where_flat = self.decode_objects(z_where, z_what)
        if return_objects:
            ret["objects"] = objects
            ret["objects_where"] = z_where_flat

        objects, depths = self.transform_objects(
            objects, z_where_flat, z_present, z_depth
        )

        reconstructions = self.reconstruct(objects, depths)
        if normalize_reconstructions:
            reconstructions = self.normalize_reconstructions(reconstructions)
        ret["reconstructions"] = reconstructions

        return ret
