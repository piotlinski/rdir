"""Tests for what encoder."""
import pytest
import torch

from src.models.components.encode.what import WhatEncoder


@pytest.mark.parametrize("latent_dim", [4, 8])
@pytest.mark.parametrize("num_hidden", [0, 2])
@pytest.mark.parametrize("scale_const", [-1.0, 0.1])
def test_what_encoder_dimensions(
    latent_dim, num_hidden, scale_const, yolov4_ints, yolov4_preds, parsed_yolov4
):
    """Verify if output shape of z_what encoder."""
    _, neck, head = parsed_yolov4
    batch_size, num_anchors, *_ = yolov4_preds[0].shape
    what_encoder = WhatEncoder(
        latent_dim=latent_dim,
        num_hidden=num_hidden,
        anchors=head.num_anchors,
        out_channels=neck.out_channels,
        scale_const=scale_const,
    )
    locs, scales = what_encoder(yolov4_ints)
    assert locs.shape == scales.shape == (batch_size, num_anchors, latent_dim)


@pytest.mark.parametrize("scale_const", [-1.0, 0.23])
def test_what_encoder_dtype(scale_const, yolov4_ints, parsed_yolov4):
    """Verify z_what encoder output dtype."""
    _, neck, head = parsed_yolov4
    what_encoder = WhatEncoder(
        latent_dim=4,
        num_hidden=1,
        anchors=head.num_anchors,
        out_channels=neck.out_channels,
        scale_const=scale_const,
    )
    locs, scales = what_encoder(yolov4_ints)
    assert locs.dtype == torch.float
    assert scales.dtype == torch.float
    assert torch.gt(scales, 0).all()


@pytest.mark.parametrize("scale_const", [0.44, 0.94])
def test_what_encoder_constant_scale(scale_const, yolov4_ints, parsed_yolov4):
    """Verify if scale is constant when set."""
    _, neck, head = parsed_yolov4
    what_encoder = WhatEncoder(
        latent_dim=4,
        num_hidden=1,
        anchors=head.num_anchors,
        out_channels=neck.out_channels,
        scale_const=scale_const,
    )
    _, scales = what_encoder(yolov4_ints)
    assert torch.all(scales == scale_const)
