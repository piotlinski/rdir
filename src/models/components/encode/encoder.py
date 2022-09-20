"""DIR encoder."""
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn

from src.models.components.encode.depth import DepthEncoder
from src.models.components.encode.heads import PresentHead, WhereHead
from src.models.components.encode.parse import parse_yolov4
from src.models.components.encode.what import WhatEncoder
from src.models.components.latents import DIRLatents


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

        self.cloned_backbone: Optional[nn.Module] = None
        self.cloned_neck: Optional[nn.Module] = None
        if clone_backbone:
            if clone_backbone == "neck":
                self.cloned_neck = deepcopy(self.neck).requires_grad_(True)
            if clone_backbone == "all":
                self.cloned_backbone = deepcopy(self.backbone).requires_grad_(True)

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

        return latents
