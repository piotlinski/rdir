"""Parsing YOLOv4 Darknet model."""
import os
import sys
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov4 import YOLOv4
from yolov4.tool.yolo_layer import YoloLayer as OriginalYOLOLayer


class HidePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_last_backbone_block(cfg_file: str) -> int:
    """Parse Darknet config to get last backbone block index."""
    block_idx = -1
    with open(cfg_file, "r") as fp:
        for line in fp:
            if "##########################" in line:
                break
            if line[0] == "[" and "net" not in line:
                block_idx += 1
    return block_idx


def get_yolo_channels(yolo: YOLOv4) -> Dict[int, int]:
    """Parse yolo config to get number of output channels in conv and route layers."""
    channels: Dict[int, int] = {}
    for idx, block in enumerate(yolo.blocks[1:]):
        if block["type"] == "convolutional":
            channels[idx] = int(block["filters"])
        elif block["type"] == "route":
            layers = block["layers"].split(",")
            layers = [int(i) if int(i) > 0 else int(i) + idx for i in layers]
            channels[idx] = sum(channels[layer] for layer in layers)
            if "groups" in block.keys() and int(block["groups"]) != 1:
                channels[idx] //= 2
        elif block["type"] in ("upsample", "maxpool", "shortcut"):
            channels[idx] = channels[idx - 1]
    return channels


class YOLOLayer(nn.Module):
    """Parsed YOLOLayer from YOLOv4."""

    def __init__(self, layer: OriginalYOLOLayer):
        """Copy parameters from original YoloLayer."""
        super().__init__()
        self.anchor_mask = layer.anchor_mask
        self.num_classes = layer.num_classes
        self.anchors = layer.anchors
        self.anchor_step = layer.anchor_step
        self.thresh = layer.thresh
        self.stride = layer.stride
        self.scale_x_y = layer.scale_x_y

        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors.extend(
                self.anchors[m * self.anchor_step : (m + 1) * self.anchor_step]
            )
        self.masked_anchors = [anchor / self.stride for anchor in masked_anchors]
        self.num_masked_anchors = len(self.anchor_mask)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        :param x: raw output of the model
        :return: tuple of:
            XYWH boxes tensor [batch, num_anchors * H * W, 1, 4]
            confidence tensor [batch, num_anchors * H * W, num_classes]
        """
        bxy, bwh, det_confs, cls_confs = self._parse_prediction(x)
        boxes = self._process_boxes(bxy, bwh)
        confs = self._process_confidences(det_confs, cls_confs)
        return boxes, confs

    def _parse_prediction(
        self, output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse YOLOLayer prediction to create boxes and confidence tensors.

        :param output: raw output of the model
        :return: tuple of:
          XY parameters [batch, num_anchors * 2, H, W]
          WH parameters [batch, num_anchors * 2, H, W]
          confidence tensor [batch, num_anchors, H, W]
          classes tensor [batch, num_anchors * num_classes, H, W]
        """
        bxy_list = []
        bwh_list = []
        det_confs_list = []
        cls_confs_list = []

        for i in range(self.num_masked_anchors):
            begin = i * (5 + self.num_classes)
            end = (i + 1) * (5 + self.num_classes)

            bxy_list.append(output[:, begin : begin + 2])
            bwh_list.append(output[:, begin + 2 : begin + 4])
            det_confs_list.append(output[:, begin + 4 : begin + 5])
            cls_confs_list.append(output[:, begin + 5 : end])

        return (
            torch.cat(bxy_list, dim=1),
            torch.cat(bwh_list, dim=1),
            torch.cat(det_confs_list, dim=1),
            torch.cat(cls_confs_list, dim=1),
        )

    def _process_confidences(
        self, det_confs: torch.Tensor, cls_confs: torch.Tensor
    ) -> torch.Tensor:
        """Process confidence tensors.

        :param det_confs: confidence tensor [batch, num_anchors, H, W]
        :param cls_confs: confidence tensor [batch, num_anchors * num_classes, H, W]
        :return: detection confidences [batch, num_anchors * H * W, num_classes]
        """
        batch, _, H, W = det_confs.shape
        det_confs = det_confs.view(batch, self.num_masked_anchors * H * W)
        cls_confs = cls_confs.view(
            batch, self.num_masked_anchors, self.num_classes, H * W
        )
        cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(
            batch, self.num_masked_anchors * H * W, self.num_classes
        )

        det_confs = torch.sigmoid(det_confs)
        cls_confs = torch.sigmoid(cls_confs)

        return cls_confs * det_confs.unsqueeze(-1)

    def _process_boxes(self, bxy: torch.Tensor, bwh: torch.Tensor) -> torch.Tensor:
        """Process box tensors.

        :param bxy: XY parameters [batch, num_anchors * 2, H, W]
        :param bwh: WH parameters [batch, num_anchors * 2, H, W]
        :return: XYWH box tensors [batch, num_anchors * H * W, 1, 4]
        """
        batch, _, H, W = bxy.shape
        bxy = torch.sigmoid(bxy) * self.scale_x_y - 0.5 * (self.scale_x_y - 1)
        bwh = torch.exp(bwh)

        grid_x = (
            torch.linspace(0, W - 1, W, device=bxy.device, dtype=bxy.dtype)
            .reshape(1, 1, 1, W)
            .repeat(1, 1, H, 1)
        )
        grid_y = (
            torch.linspace(0, H - 1, H, device=bxy.device, dtype=bxy.dtype)
            .reshape(1, 1, H, 1)
            .repeat(1, 1, 1, W)
        )

        anchor_w = []
        anchor_h = []
        for i in range(self.num_masked_anchors):
            anchor_w.append(self.masked_anchors[i * 2])
            anchor_h.append(self.masked_anchors[i * 2 + 1])

        bx_list = []
        by_list = []
        bw_list = []
        bh_list = []

        for i in range(self.num_masked_anchors):
            ii = i * 2
            bx_list.append(bxy[:, ii : ii + 1] + grid_x)
            by_list.append(bxy[:, ii + 1 : ii + 2] + grid_y)
            bw_list.append(bwh[:, ii : ii + 1] * anchor_w[i])
            bh_list.append(bwh[:, ii + 1 : ii + 2] * anchor_h[i])

        bx = (
            torch.cat(bx_list, dim=1).view(batch, self.num_masked_anchors * H * W, 1)
            / W
        )
        by = (
            torch.cat(by_list, dim=1).view(batch, self.num_masked_anchors * H * W, 1)
            / H
        )
        bw = (
            torch.cat(bw_list, dim=1).view(batch, self.num_masked_anchors * H * W, 1)
            / W
        )
        bh = (
            torch.cat(bh_list, dim=1).view(batch, self.num_masked_anchors * H * W, 1)
            / H
        )

        return torch.cat((bx, by, bw, bh), dim=2).view(
            batch, self.num_masked_anchors * H * W, 1, 4
        )


class YOLOModule(nn.Module):
    """Meta-module for YOLO-based modules."""

    def __init__(self, yolo: YOLOv4, start: int = 0, end: int = -1):
        """Initialize YOLO-based module.

        :param yolo: YOLOv4 Darknet model
        :param start: start index to include
        :param end: end index to include
        """
        super().__init__()
        end = end if end > 0 else len(yolo.models)
        self.models = yolo.models[start:end]
        self._cfg = yolo.blocks[start + 1 : end + 1]
        self._start_idx = start
        self._end_idx = end
        self.keep = self._get_keep(yolo)
        self.drop_indices = self._get_drop_indices(yolo)
        for idx in self.drop_indices:
            self.models[idx - self._start_idx] = None  # un-refer unused modules
        self.out_channels: Dict[int, int] = get_yolo_channels(yolo)

    def _get_keep(self, yolo: YOLOv4) -> Dict[int, int]:
        """Get indices and counts of blocks outputs to keep."""
        indices = []
        for idx, block in enumerate(yolo.blocks[1:]):
            if block["type"] == "route":
                for i in block["layers"].split(","):
                    indices.append(int(i) if int(i) > 0 else int(i) + idx)
            elif block["type"] == "shortcut":
                i = int(block["from"])
                indices.append(i if i > 0 else i + idx)
                indices.append(idx - 1)
        keep: Dict[int, int] = defaultdict(int)
        for idx in indices:
            if self._start_idx <= idx < self._end_idx:
                keep[idx] += 1
        return keep

    def _get_drop_indices(self, yolo: YOLOv4) -> Set[int]:
        """Get indices of blocks that should be excluded from processing."""
        return set()

    def forward(
        self, x: Union[torch.Tensor, Dict[int, torch.Tensor]]
    ) -> Dict[int, torch.Tensor]:
        """Forward pass."""
        intermediates: Dict[int, torch.Tensor] = {}
        output_indices = self.keep.copy()
        if not isinstance(x, torch.Tensor):
            intermediates.update(x)
            x = intermediates[self._start_idx - 1]
        for idx, block in enumerate(self._cfg, start=self._start_idx):
            abs_idx = idx - self._start_idx
            if idx in self.drop_indices:
                if idx - 1 not in self.drop_indices:
                    intermediates[idx - 1] = x
                    output_indices[idx - 1] += 1
                continue
            if block["type"] in [
                "convolutional",
                "maxpool",
                "reorg",
                "upsample",
                "avgpool",
                "softmax",
                "connected",
            ]:
                x = self.models[abs_idx](x)
            elif block["type"] == "route":
                layers = block["layers"].split(",")
                layers = [int(i) if int(i) > 0 else int(i) + idx for i in layers]
                if len(layers) == 1:
                    x = intermediates[layers[0]]
                    output_indices[layers[0]] -= 1
                    if "groups" in block.keys() and int(block["groups"]) != 1:
                        groups = int(block["groups"])
                        group_id = int(block["group_id"])
                        _, b, _, _ = x.shape
                        x = x[:, b // groups * group_id : b // groups * (group_id + 1)]
                elif len(layers) == 2:
                    x1 = intermediates[layers[0]]
                    output_indices[layers[0]] -= 1
                    x2 = intermediates[layers[1]]
                    output_indices[layers[1]] -= 1
                    x = torch.cat((x1, x2), 1)
                elif len(layers) == 4:
                    x1 = intermediates[layers[0]]
                    output_indices[layers[0]] -= 1
                    x2 = intermediates[layers[1]]
                    output_indices[layers[1]] -= 1
                    x3 = intermediates[layers[2]]
                    output_indices[layers[2]] -= 1
                    x4 = intermediates[layers[3]]
                    output_indices[layers[3]] -= 1
                    x = torch.cat((x1, x2, x3, x4), 1)
                else:
                    raise ValueError("Wrong number of layers")
            elif block["type"] == "shortcut":
                from_layer = int(block["from"])
                activation = block["activation"]
                from_layer = from_layer if from_layer > 0 else from_layer + idx
                x1 = intermediates[from_layer]
                output_indices[from_layer] -= 1
                x2 = intermediates[idx - 1]
                output_indices[idx - 1] -= 1
                x = x1 + x2
                if activation == "leaky":
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == "relu":
                    x = F.relu(x, inplace=True)
            else:
                raise ValueError("Incorrect model type for backbone")
            if idx in self.keep:
                intermediates[idx] = x
        return {
            key: intermediates[key]
            for key, value in output_indices.items()
            if value > 0
        }


class Backbone(YOLOModule):
    """CSPDarknet53 backbone extracted from YOLOv4."""

    def __init__(self, yolo: YOLOv4, last_backbone_block: int):
        """Initialize backbone."""
        self.last_backbone_block = last_backbone_block
        super().__init__(yolo, start=0, end=self.last_backbone_block + 1)

    def _get_keep(self, yolo: YOLOv4) -> Dict[int, int]:
        """Include last backbone block in keep indices."""
        keep = super()._get_keep(yolo)
        keep[self.last_backbone_block] += 1
        return keep


class Neck(YOLOModule):
    """PANet neck extracted from YOLOv4."""

    def __init__(self, yolo: YOLOv4, last_backbone_block: int):
        """Initialize neck."""
        super().__init__(yolo, start=last_backbone_block + 1, end=-1)

    def _get_drop_indices(self, yolo: YOLOv4) -> Set[int]:
        """Get indices of blocks that should be excluded from processing."""
        indices = set()
        for idx, block in enumerate(yolo.blocks[1:]):
            if block["type"] == "yolo":
                indices.update({idx - 2, idx - 1, idx})
        return indices


class Head(nn.Module):
    """YOLOv4 head extracted from YOLOv4."""

    def __init__(self, yolo: YOLOv4):
        """Initialize head."""
        super().__init__()
        self.num_anchors: Dict[int, int] = {}
        self._get_heads(yolo)

    def _get_heads(self, yolo: YOLOv4):
        """Add YOLO heads to the module."""
        for idx, block in enumerate(yolo.blocks[1:]):
            if block["type"] == "yolo":
                conv_1, conv2, yolo_layer = yolo.models[idx - 2 : idx + 1]
                parsed_yolo_layer = YOLOLayer(yolo_layer)
                head = nn.Sequential(conv_1, conv2, parsed_yolo_layer)
                self.add_module(f"head_{idx - 3}", head)
                self.num_anchors[idx - 3] = parsed_yolo_layer.num_masked_anchors

    def forward(self, x: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        boxes, confs = [], []
        for idx, intermediate in x.items():
            head_boxes, head_confs = self.get_submodule(f"head_{idx}")(intermediate)
            boxes.append(head_boxes)
            confs.append(head_confs)

        return torch.cat(boxes, dim=1), torch.cat(confs, dim=1)


def parse_yolov4(
    cfg_file: str, weights_file: Optional[str] = None
) -> Tuple[Backbone, Neck, Head]:
    """Load and parse YOLOv4 model."""
    with HidePrint():
        yolo = YOLOv4(cfg_file)
    last_backbone_block = get_last_backbone_block(cfg_file)
    if weights_file is not None:
        yolo.load_weights(weights_file)
    return (
        Backbone(yolo, last_backbone_block),
        Neck(yolo, last_backbone_block),
        Head(yolo),
    )
