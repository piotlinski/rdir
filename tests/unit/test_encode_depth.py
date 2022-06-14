"""Tests for depth encoder."""
import pytest
import torch

from src.models.components.encode.depth import DepthEncoder


@pytest.mark.parametrize("scale_const", [-1.0, 0.2])
def test_depth_encoder_dimensions(
    scale_const, yolov4_ints, yolov4_preds, parsed_yolov4
):
    """Verify output shape of z_depth encoder."""
    _, neck, head = parsed_yolov4
    batch_size, num_anchors, *_ = yolov4_preds[0].shape
    depth_encoder = DepthEncoder(
        anchors=head.num_anchors,
        out_channels=neck.out_channels,
        scale_const=scale_const,
    )
    locs, scales = depth_encoder(yolov4_ints)
    assert locs.shape == scales.shape == (batch_size, num_anchors, 1)


@pytest.mark.parametrize("scale_const", [-1.0, 0.64])
def test_depth_encoder_dtype(scale_const, yolov4_ints, parsed_yolov4):
    """Verify z_depth encoder output dtype."""
    _, neck, head = parsed_yolov4
    depth_encoder = DepthEncoder(
        anchors=head.num_anchors,
        out_channels=neck.out_channels,
        scale_const=scale_const,
    )
    locs, scales = depth_encoder(yolov4_ints)
    assert locs.dtype == torch.float
    assert scales.dtype == torch.float
    assert torch.gt(scales, 0).all()


@pytest.mark.parametrize("scale_const", [0.11, 0.75])
def test_depth_encoder_constant_scale(scale_const, yolov4_ints, parsed_yolov4):
    """Verify if scale is constant when set."""
    _, neck, head = parsed_yolov4
    depth_encoder = DepthEncoder(
        anchors=head.num_anchors,
        out_channels=neck.out_channels,
        scale_const=scale_const,
    )
    _, scales = depth_encoder(yolov4_ints)
    assert torch.all(scales == scale_const)
