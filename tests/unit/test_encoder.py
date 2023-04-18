"""Test DIR encoder."""
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.models.components.encode.encoder import Encoder, RNNEncoder


@pytest.mark.parametrize("z_what_size", [4, 8, 16])
@patch("src.models.components.encode.encoder.parse_yolov4")
def test_encoder_dimensions(
    parse_yolov4_mock, z_what_size, parsed_yolov4, sample_inputs
):
    """Verify encoder output dimensions."""
    batch_size = sample_inputs.shape[0]
    parse_yolov4_mock.return_value = parsed_yolov4
    encoder = Encoder(yolo=("test", None), z_what_size=z_what_size)
    (
        z_where,
        z_present,
        (z_what_loc, z_what_scale),
        (z_depth_loc, z_depth_scale),
    ) = encoder(sample_inputs)
    num_anchors = z_where.shape[1]
    assert z_where.shape == (batch_size, num_anchors, 4)
    assert z_present.shape == (batch_size, num_anchors, 1)
    assert (
        z_what_loc.shape == z_what_scale.shape == (batch_size, num_anchors, z_what_size)
    )
    assert z_depth_loc.shape == z_depth_scale.shape == (batch_size, num_anchors, 1)


@pytest.mark.parametrize(
    "modules_enabled",
    [
        [False, False, False, False, False],
        [False, False, False, True, True],
        [False, False, True, True, True],
        [False, True, True, True, True],
        [True, True, True, True, True],
        [True, False, True, True, True],
        [True, True, False, True, True],
        [True, False, False, True, True],
    ],
)
@patch("src.models.components.encode.encoder.parse_yolov4")
def test_encoder_train(parse_yolov4_mock, modules_enabled, parsed_yolov4):
    """Verify if training submodules can be disabled."""
    kwargs_keys = [
        "train_backbone",
        "train_neck",
        "train_head",
        "train_what",
        "train_depth",
    ]
    module_names = ["backbone", "neck", "head", "what_enc", "depth_enc"]
    parse_yolov4_mock.return_value = parsed_yolov4
    encoder = Encoder(yolo=("test", None), **dict(zip(kwargs_keys, modules_enabled)))
    for name, requires_grad in zip(module_names, modules_enabled):
        assert all(
            param.requires_grad == requires_grad
            for param in getattr(encoder, name).parameters()
        )


@pytest.mark.parametrize("z_what_size", [4, 8, 16])
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
@patch("src.models.components.encode.encoder.parse_yolov4")
def test_rnn_encoder_dimensions(
    parse_yolov4_mock, z_what_size, kernel_size, parsed_yolov4, sample_inputs
):
    """Verify rnn encoder dimensions."""
    parse_yolov4_mock.return_value = parsed_yolov4
    inputs = nn.utils.rnn.PackedSequence(
        sample_inputs.expand(2, -1, -1, -1), torch.tensor([1, 1])
    )
    encoder = Encoder(yolo=("test", None), z_what_size=z_what_size)
    rnn_encoder = RNNEncoder(encoder, n_rnn_hidden=2, rnn_kernel_size=kernel_size)

    (
        z_where,
        z_present,
        (z_what_loc, z_what_scale),
        (z_depth_loc, z_depth_scale),
    ) = rnn_encoder(inputs)

    n_objects = z_where.data.shape[1]
    assert z_where.data.shape == (2, n_objects, 4)
    assert z_present.data.shape == (2, n_objects, 1)
    assert (
        z_what_loc.data.shape == z_what_scale.data.shape == (2, n_objects, z_what_size)
    )
    assert z_depth_loc.data.shape == z_depth_scale.data.shape == (2, n_objects, 1)
