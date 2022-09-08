"""DIR encoder."""
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.components.encode.depth import DepthEncoder
from src.models.components.encode.heads import PresentHead, WhereHead
from src.models.components.encode.parse import parse_yolov4
from src.models.components.encode.what import WhatEncoder

DIRLatents = Tuple[
    torch.Tensor,
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]


class Encoder(nn.Module):
    """Module encoding input image to latent representation."""

    def __init__(
        self,
        yolo_cfg_file: str,
        yolo_weights_file: Optional[str] = None,
        z_what_size: int = 64,
        z_what_hidden: int = 1,
        z_what_scale_const: float = -1.0,
        z_depth_scale_const: float = -1.0,
        reset_non_present: bool = False,
        train_backbone: bool = False,
        train_neck: bool = False,
        train_head: bool = False,
        train_what: bool = True,
        train_depth: bool = True,
        clone_backbone: str = "",
    ):
        """
        :param yolo_cfg_file: path to yolov4 config file
        :param yolo_weights_file: path to pretrained yolov4 model
        :param z_what_size: z_what latent representation size
        :param z_what_hidden: hidden layers in z_what encoders
        :param z_what_scale_const: allows to set z_what scale to a constant
            (negative for trainable, 0 for deterministic z_what)
        :param z_depth_scale_const: allows to set z_depth scale to a constant
            (negative for trainable, 0 for deterministic z_depth)
        :param reset_non_present: set non-present latents to some ordinary ones
        :param train_backbone: perform backbone training
        :param train_neck: perform neck training
        :param train_head: perform head training
        :param train_what: perform z_what encoder training
        :param train_depth: perform z_depth encoder training
        :param clone_backbone: clone backbone (all or neck) for what and depth encoders
        """
        super().__init__()

        self.backbone, self.neck, self.head = parse_yolov4(
            cfg_file=yolo_cfg_file, weights_file=yolo_weights_file
        )
        self.backbone.requires_grad_(train_backbone)
        self.neck.requires_grad_(train_neck)
        self.head.requires_grad_(train_head)

        self.where_head = WhereHead()
        self.present_head = PresentHead()

        self.what_enc = WhatEncoder(
            latent_dim=z_what_size,
            num_hidden=z_what_hidden,
            anchors=self.head.num_anchors,
            out_channels=self.neck.out_channels,
            scale_const=z_what_scale_const,
        ).requires_grad_(train_what)
        self.depth_enc = DepthEncoder(
            anchors=self.head.num_anchors,
            out_channels=self.neck.out_channels,
            scale_const=z_depth_scale_const,
        ).requires_grad_(train_depth)

        self._reset_non_present = reset_non_present
        self._z_present_eps = 1e-3
        self.register_buffer("_empty_loc", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer("_empty_scale", torch.tensor(1.0, dtype=torch.float))

        self.cloned_backbone: Optional[nn.Module] = None
        self.cloned_neck: Optional[nn.Module] = None
        if clone_backbone:
            if clone_backbone == "neck":
                self.cloned_neck = deepcopy(self.neck).requires_grad_(True)
            if clone_backbone == "all":
                self.cloned_backbone = deepcopy(self.backbone).requires_grad_(True)

    def reset_non_present(self, latents: DIRLatents) -> DIRLatents:
        """Reset latents, whose z_present is 0."""
        (
            z_where,
            z_present,
            (z_what_loc, z_what_scale),
            (z_depth_loc, z_depth_scale),
        ) = latents
        present_mask = torch.gt(z_present, self._z_present_eps)
        z_what_loc = torch.where(
            present_mask, z_what_loc, self._empty_loc.type(z_what_loc.dtype)
        )
        if self.what_enc.is_probabilistic:
            z_what_scale = torch.where(
                present_mask, z_what_scale, self._empty_scale.type(z_what_scale.dtype)
            )
        z_depth_loc = torch.where(
            present_mask, z_depth_loc, self._empty_loc.type(z_depth_loc.dtype)
        )
        if self.depth_enc.is_probabilistic:
            z_depth_scale = torch.where(
                present_mask, z_depth_scale, self._empty_scale.type(z_depth_scale.dtype)
            )
        return (
            z_where,
            z_present,
            (z_what_loc, z_what_scale),
            (z_depth_loc, z_depth_scale),
        )

    def forward(self, images: torch.Tensor) -> DIRLatents:
        """Encode images to latent representation.

        :param images: image tensor (batch size x channels x image_size x image_size)
        :returns: latent representation
            (z_where, z_present, z_what (loc & scale), z_depth (loc & scale))
        """
        features = self.backbone(images)
        intermediates = self.neck(features)

        boxes, confs = self.head(intermediates)
        z_where = self.where_head(boxes)
        z_present = self.present_head(confs)

        if self.cloned_backbone is not None:
            features = self.cloned_backbone(images)
        if self.cloned_neck is not None:
            intermediates = self.cloned_neck(features)

        z_what = self.what_enc(intermediates)
        z_depth = self.depth_enc(intermediates)

        latents = (z_where, z_present, z_what, z_depth)
        if self._reset_non_present:
            latents = self.reset_non_present(latents)

        return latents
