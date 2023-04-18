"""Latents handler for DIR."""
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

Tensor = Union[torch.Tensor, nn.utils.rnn.PackedSequence]

DIRLatents = Tuple[Tensor, Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]
DIRRepresentation = Tuple[Tensor, Tensor, Tensor, Tensor]


class LatentHandler(nn.Module):
    """Module for handling latents."""

    def __init__(
        self,
        reset_non_present: bool = False,
        negative_percentage: float = 0.1,
        max_objects: Optional[int] = 10,
    ):
        """
        :param reset_non_present: set non-present latents to some ordinary ones
        :param negative_percentage: percentage of negative objects to be added
        :param max_objects: maximum number of objects to be present in the image
            (if None, then no limit is applied)
        """
        super().__init__()

        self._reset_non_present = reset_non_present
        self._negative_percentage = negative_percentage
        self._max_objects = max_objects

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

    @staticmethod
    def limit_positive(
        sampled_present: torch.Tensor, z_present: torch.Tensor, max_objects: int
    ) -> torch.Tensor:
        """Limit number of positive objects."""
        n_present = torch.sum(sampled_present, dim=1, dtype=torch.long)
        present_mask = sampled_present > 0
        for idx, n in enumerate(n_present):
            if n > max_objects:
                mask = present_mask[idx]
                positive = z_present[idx][mask]
                indices = torch.arange(
                    mask.shape[0], dtype=torch.long, device=mask.device
                ).unsqueeze(-1)[mask]

                _, sort_indices = torch.sort(positive)
                to_drop = indices[sort_indices][: n - max_objects]

                sampled_present[(idx, to_drop)] = 0

        return sampled_present

    @staticmethod
    def negative_indices(z_present: torch.Tensor, padded_size: int) -> torch.Tensor:
        """Get array of zero-valued indices.

        .. note: first objects' index will be used as padding     we use -1 to
        indicate indices where present objects are
        """
        zero = torch.nonzero(torch.eq(z_present, 0))
        shuffled = torch.randperm(zero.shape[0])
        zero = zero[shuffled]

        indices = []
        for idx, n in enumerate(torch.sum(z_present, dim=1, dtype=torch.long)):
            row = zero[zero[:, 0] == idx][:, 1]
            row = F.pad(row, (padded_size - len(row), 0), value=row[0])
            row[:n] = -1
            indices.append(row)

        return torch.stack(indices)

    def add_negative(self, z_present: torch.Tensor) -> torch.Tensor:
        """Add some present objects for learning negative as well.

        .. note: this method ensures that we have equal number of objects for
        each image we indicate negative objects by putting -1 in z_present
        """
        n_present = torch.sum(z_present, dim=1, dtype=torch.long)
        max_present = torch.max(n_present)
        negative_added = int(self._negative_percentage * z_present.shape[1])
        n_objects = max_present + negative_added

        added = self.negative_indices(z_present, n_objects)

        x_mask, y_mask = torch.nonzero(added != -1, as_tuple=True)
        modified = z_present.clone()
        modified[(x_mask, added[(x_mask, y_mask)])] = -1

        return modified

    @staticmethod
    def filter(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Filter representation based on mask."""
        return tensor[mask.expand_as(tensor)].view(
            tensor.shape[-0], -1, tensor.shape[-1]
        )

    def filter_representation(
        self, representation: DIRRepresentation
    ) -> DIRRepresentation:
        """Filter representation according to z_present."""
        z_where, z_present, z_what, z_depth = representation
        z_present = self.add_negative(z_present)
        present_mask = torch.ne(z_present, 0)
        z_where = self.filter(z_where, present_mask)
        z_present = self.filter(z_present, present_mask)
        z_what = self.filter(z_what, present_mask)
        z_depth = self.filter(z_depth, present_mask)
        return z_where, z_present, z_what, z_depth

    def forward(
        self,
        latents: DIRLatents,
        where_fn: Callable[[torch.Tensor], torch.Tensor],
        present_fn: Callable[[torch.Tensor], torch.Tensor],
        what_fn: Callable[[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        depth_fn: Callable[[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    ) -> DIRRepresentation:
        if self._reset_non_present:
            latents = self.reset_non_present(latents)
        z_where, z_present, z_what, z_depth = latents

        sampled_where = where_fn(z_where)
        sampled_present = present_fn(z_present)
        sampled_what = what_fn(z_what)
        sampled_depth = depth_fn(z_depth)

        if self._max_objects is not None:
            sampled_present = self.limit_positive(
                sampled_present, z_present, self._max_objects
            )

        representation = (
            sampled_where,
            sampled_present,
            sampled_what,
            sampled_depth,
        )

        filtered = self.filter_representation(representation)

        return filtered
