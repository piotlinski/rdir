"""Tests for YOLOv4 parsing."""
from unittest.mock import mock_open, patch

import pytest
import torch

from src.models.components.encode.parse import (
    Backbone,
    Head,
    HidePrint,
    Neck,
    YOLOLayer,
    get_last_backbone_block,
    get_yolo_channels,
    parse_yolov4,
)


@pytest.fixture(scope="session")
def yololayer(yolov4):
    """Sample YOLOv4 YoloLayer."""
    return yolov4.models[-1]


@pytest.fixture(scope="session")
def masked_anchors(yololayer):
    """Masked anchors generated from yololayer."""
    masked_anchors = []
    for m in yololayer.anchor_mask:
        masked_anchors.extend(
            yololayer.anchors[
                m * yololayer.anchor_step : (m + 1) * yololayer.anchor_step
            ]
        )
    return [anchor / yololayer.stride for anchor in masked_anchors]


@pytest.fixture(scope="session")
def parsed_yololayer(yololayer):
    """Parsed YOLOLayer."""
    return YOLOLayer(yololayer)


def xywh_to_x1y1x2y2(xywh: torch.Tensor) -> torch.Tensor:
    """Convert YOLOv4 xywh to x1y1x2y2."""
    x = xywh[..., [0]]
    y = xywh[..., [1]]
    w = xywh[..., [2]]
    h = xywh[..., [3]]
    x1 = x - w * 0.5
    y1 = y - h * 0.5
    x2 = x1 + w
    y2 = y1 + h
    return torch.cat((x1, y1, x2, y2), dim=-1)


def test_hide_print(capsys):
    """Test if HidePrint stops printing output."""
    with HidePrint():
        print("This should not be printed.")
    printed = capsys.readouterr()
    assert printed.out == ""


def test_print_after_hide_print(capsys):
    """Test if printing is restored outside HidePrint."""
    with HidePrint():
        print("This should not be printed.")
    print("This should be printed.")
    printed = capsys.readouterr()
    assert printed.out == "This should be printed.\n"


@pytest.mark.parametrize("n_pre_last, n_post_last", [(1, 1), (2, 1), (3, 2)])
def test_get_last_backbone_block(n_pre_last, n_post_last):
    """Check if last backbone block is selected properly."""
    data = "[net]\nabc\n"
    for _ in range(n_pre_last):
        data += "[block before]\ndef\n"
    data += "##########################\n"
    for _ in range(n_post_last):
        data += "[block after]\nghi\n"

    with patch("builtins.open", mock_open(read_data=data)):
        assert get_last_backbone_block("any/path") == n_pre_last - 1


def test_get_yolo_channels(yolov4, yolov4_feats, yolov4_ints):
    """Verify if yolo tensor channels are fetched correctly."""
    out_channels = get_yolo_channels(yolov4)
    for key, output in yolov4_feats.items():
        assert out_channels[key] == output.shape[1]
    for key, output in yolov4_ints.items():
        assert out_channels[key] == output.shape[1]


def test_yololayer(yololayer, parsed_yololayer, masked_anchors):
    """Verify if YOLOLayer fields match those of the Original YOLO Layer."""
    assert parsed_yololayer.anchor_mask == yololayer.anchor_mask
    assert parsed_yololayer.num_classes == yololayer.num_classes
    assert parsed_yololayer.anchors == yololayer.anchors
    assert parsed_yololayer.thresh == yololayer.thresh
    assert parsed_yololayer.stride == yololayer.stride
    assert parsed_yololayer.scale_x_y == yololayer.scale_x_y

    assert parsed_yololayer.num_masked_anchors == len(yololayer.anchor_mask)
    assert parsed_yololayer.masked_anchors == masked_anchors


@pytest.mark.parametrize("seed", [7, 13])
@pytest.mark.parametrize("batch", [1, 3])
def test_yololayer_forward(batch, seed, yololayer, parsed_yololayer):
    """Verify if YOLOLayer output matches Original YOLO Layer's."""
    generator = torch.random.manual_seed(seed)
    inputs = torch.rand(batch, 45, 13, 13, generator=generator)
    expected_boxes, expected_confs = yololayer.eval()(inputs)
    results_boxes_xywh, results_confs = parsed_yololayer.eval()(inputs)

    results_boxes_x1y1x2y2 = xywh_to_x1y1x2y2(results_boxes_xywh)

    assert torch.allclose(expected_boxes, results_boxes_x1y1x2y2)
    assert torch.allclose(expected_confs, results_confs)


@pytest.mark.parametrize("last_backbone_block", [10, 20, 30])
def test_backbone(last_backbone_block, yolov4):
    """Verify if backbone is parsed correctly."""
    backbone = Backbone(yolov4, last_backbone_block)
    assert len(backbone.models) == last_backbone_block + 1


@pytest.mark.parametrize("last_backbone_block", [10, 20, 30])
def test_neck(last_backbone_block, yolov4):
    """Verify if neck is parsed correctly."""
    neck = Neck(yolov4, last_backbone_block)
    assert len(neck.models) == len(yolov4.models) - last_backbone_block - 1


def test_head(yolov4):
    """Verify if head is parsed correctly."""
    head = Head(yolov4)
    for child in head.children():
        assert len(child) == 3
        assert isinstance(child[-1], YOLOLayer)


@pytest.mark.parametrize("seed", [7, 13])
@pytest.mark.parametrize("batch", [1, 3])
@patch("src.models.components.encode.parse.YOLOv4")
def test_parsed_output(yolov4_mock, batch, seed, yolov4, yolov4_cfg):
    """Verify if parsed model output matches the original."""
    generator = torch.random.manual_seed(seed)
    inputs = torch.rand(batch, 3, 32, 32, generator=generator)
    yolov4_mock.return_value = yolov4.eval()
    with patch("builtins.open", mock_open(read_data=yolov4_cfg)):
        backbone, neck, head = parse_yolov4("test")
    with torch.no_grad():
        features = backbone(inputs)
        intermediates = neck(features)
        results_boxes_xywh, results_confs = head(intermediates)
        results_boxes_x1y1x2y2 = xywh_to_x1y1x2y2(results_boxes_xywh)

        expected_boxes, expected_confs = yolov4(inputs)

    assert torch.allclose(expected_boxes, results_boxes_x1y1x2y2)
    assert torch.allclose(expected_confs, results_confs)
