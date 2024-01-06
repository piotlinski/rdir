"""DIR encoder."""
from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import ops

from src.models.components.encode.depth import DepthEncoder
from src.models.components.encode.heads import PresentHead, WhereHead
from src.models.components.encode.mix import Downscaler, Mixer, NoMixer
from src.models.components.encode.parse import Backbone, Head, Neck, parse_yolov4
from src.models.components.encode.rnn import SeqEncoder, packed_forward
from src.models.components.encode.what import WhatEncoder
from src.models.components.latents import DIRLatents


class Encoder(nn.Module):
    """Module encoding input image to latent representation."""

    MIXERS = {"mixer": Mixer, "downscaler": Downscaler, "none": NoMixer}

    def __init__(
        self,
        yolo: Union[Tuple[Backbone, Neck, Head], Tuple[str, Optional[str]]],
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
        what_enc: Optional[WhatEncoder] = None,
        depth_enc: Optional[DepthEncoder] = None,
        cloned_neck: Optional[Neck] = None,
        cloned_backbone: Optional[Backbone] = None,
        filter_classes: Optional[Tuple[int, ...]] = None,
        nms_threshold: float = 0.45,
        nms_always: bool = False,
        mixer: str = "mixer",
    ):
        """
        :param yolo: path to yolov4 config file and optional weights
            or tuple of backbone, neck and head
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
        :param what_enc: what encoder from a trained model
        :param depth_enc: depth encoder from a trained model
        :param cloned_neck: neck from a trained model
        :param cloned_backbone: backbone from a trained model
        :param filter_classes: filter classes from the prediction
        :param nms_threshold: non-maximum suppression threshold
        :param nms_always: run NMS on train and val
        :param mixer: mixer type (mixer, downscaler or none)
        """
        super().__init__()

        self.clone_backbone = clone_backbone
        try:
            self.backbone, self.neck, self.head = yolo  # type: ignore
        except ValueError:
            yolo_cfg_file, yolo_weights_file = yolo  # type: ignore
            self.backbone, self.neck, self.head = parse_yolov4(
                cfg_file=yolo_cfg_file, weights_file=yolo_weights_file
            )
        self.backbone.requires_grad_(train_backbone)
        self.neck.requires_grad_(train_neck)
        self.head.requires_grad_(train_head)

        self.where_head = WhereHead()
        self.present_head = PresentHead(filter_classes)

        self.mixer = self.MIXERS[mixer](self.head.num_anchors, self.neck.out_channels)

        self.what_enc = what_enc or WhatEncoder(
            latent_dim=z_what_size,
            num_hidden=z_what_hidden,
            anchors=self.head.num_anchors,
            out_channels=self.mixer.out_channels,
            scale_const=z_what_scale_const,
        )
        self.what_enc.requires_grad_(train_what)
        self.depth_enc = depth_enc or DepthEncoder(
            anchors=self.head.num_anchors,
            out_channels=self.mixer.out_channels,
            scale_const=z_depth_scale_const,
        )
        self.depth_enc.requires_grad_(train_depth)

        self.cloned_backbone: Optional[nn.Module] = cloned_backbone
        self.cloned_neck: Optional[nn.Module] = cloned_neck
        if clone_backbone:
            if clone_backbone == "neck":
                self.cloned_neck = cloned_neck or deepcopy(self.neck)
                self.cloned_neck.requires_grad_(True)
            if clone_backbone == "all":
                self.cloned_backbone = cloned_backbone or deepcopy(self.backbone)
                self.cloned_backbone.requires_grad_(True)

        self.nms_threshold = nms_threshold
        self.nms_always = nms_always

    @staticmethod
    def xywh_to_x1y1x2y2(boxes: torch.Tensor) -> torch.Tensor:
        """Convert xywh boxes to x1y1x2y2 boxes."""
        x1y1 = boxes[..., :2] - boxes[..., 2:] / 2
        x2y2 = boxes[..., :2] + boxes[..., 2:] / 2
        return torch.cat((x1y1, x2y2), dim=-1)

    def run_nms(
        self, where_and_present: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Run batched non-maximum suppression on latents."""
        z_where, z_present = where_and_present
        batch_size, n_anchors, _ = z_where.shape
        boxes = self.xywh_to_x1y1x2y2(z_where)
        indices = torch.arange(batch_size, device=z_where.device)
        indices = indices[:, None].expand(batch_size, n_anchors).flatten()

        boxes_flat = boxes.flatten(0, 1)
        scores_flat = z_present.flatten()
        indices_flat = ops.boxes.batched_nms(
            boxes_flat, scores_flat, indices, self.nms_threshold
        )

        mask = torch.zeros_like(z_present).flatten()
        mask[indices_flat] = 1
        mask = mask.view_as(z_present)
        return z_present * mask

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

        if self.nms_always or not self.training:
            z_present = self.run_nms((z_where, z_present))

        if self.cloned_backbone is not None:
            features = self.cloned_backbone(images)
        if self.cloned_neck is not None:
            intermediates = self.cloned_neck(features)

        intermediates = self.mixer(intermediates)

        z_what = self.what_enc(intermediates)
        z_depth = self.depth_enc(intermediates)

        latents = (z_where, z_present, z_what, z_depth)

        return latents


class RNNEncoder(nn.Module):
    """Sequential Encoder."""

    def __init__(
        self,
        encoder: Encoder,
        n_rnn_hidden: int = 2,
        rnn_kernel_size: int = 5,
        rnn_cls: str = "gru",
        n_rnn_cells: int = 2,
        rnn_bidirectional: bool = False,
        train_rnn: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.seq_enc = SeqEncoder(
            anchors=self.encoder.head.num_anchors,
            out_channels=self.encoder.mixer.out_channels,
            num_hidden=n_rnn_hidden,
            kernel_size=rnn_kernel_size,
            rnn_cls=rnn_cls,
            n_cells=n_rnn_cells,
            bidirectional=rnn_bidirectional,
        ).requires_grad_(train_rnn)
        self.seq_mixer = self.encoder.mixer.__class__(
            self.encoder.head.num_anchors, self.encoder.mixer.out_channels
        )

    def forward(
        self, images: nn.utils.rnn.PackedSequence, inject_hidden: bool = False
    ) -> DIRLatents:
        """Encode images sequentially."""
        features = packed_forward(self.encoder.backbone, images)
        intermediates = packed_forward(self.encoder.neck, features)

        boxes, confs = packed_forward(self.encoder.head, intermediates)
        z_where = packed_forward(self.encoder.where_head, boxes)
        z_present = packed_forward(self.encoder.present_head, confs)

        if self.encoder.nms_always or not self.training:
            z_present = packed_forward(self.encoder.run_nms, (z_where, z_present))

        if self.encoder.cloned_backbone is not None:
            features = packed_forward(self.encoder.cloned_backbone, images)
        if self.encoder.cloned_neck is not None:
            intermediates = packed_forward(self.encoder.cloned_neck, features)

        intermediates = packed_forward(self.encoder.mixer, intermediates)
        intermediates = self.seq_enc(intermediates, inject_hidden)
        intermediates = packed_forward(self.seq_mixer, intermediates)

        z_what = packed_forward(self.encoder.what_enc, intermediates)
        z_depth = packed_forward(self.encoder.depth_enc, intermediates)

        latents = (z_where, z_present, z_what, z_depth)

        return latents
