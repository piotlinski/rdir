"""Test recurrent encoder."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from src.models.components.encode.rnn import (
    SeqEncoder,
    SeqRNN,
    packed_forward,
)


@pytest.fixture
def sample_packed_sequence():
    """Sample packed sequence."""
    data = torch.rand(3, 16, 4, 4)
    batch_sizes = torch.tensor([2, 1])
    return nn.utils.rnn.PackedSequence(data, batch_sizes)


@pytest.mark.parametrize("out_channels", [4, 8])
def test_packed_forward(out_channels, sample_packed_sequence):
    """Test running forward on packed data with non-recurrent module."""
    batch_size, in_channels, *size = sample_packed_sequence.data.shape
    test_module = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    output = packed_forward(test_module, sample_packed_sequence)

    assert output.data.shape == (batch_size, out_channels, *size)
    torch.testing.assert_allclose(
        output.batch_sizes, sample_packed_sequence.batch_sizes
    )


def test_preprocess_sequence(sample_packed_sequence):
    """Verify if sequence is converted to appropriate format."""
    preprocessed, permuted_shape = SeqRNN.preprocess_sequence(sample_packed_sequence)

    assert preprocessed.data.shape == (48, 16)
    torch.testing.assert_allclose(preprocessed.batch_sizes, torch.tensor([32, 16]))
    assert permuted_shape == (3, 4, 4, 16)


def test_postprocess_sequence(sample_packed_sequence):
    """Verify if processed sequence is restored to original shape."""
    preprocessed, permuted_shape = SeqRNN.preprocess_sequence(sample_packed_sequence)

    postprocessed = SeqRNN.postprocess_sequence(preprocessed, permuted_shape)

    assert postprocessed.data.shape == sample_packed_sequence.data.shape
    torch.testing.assert_allclose(postprocessed.data, sample_packed_sequence.data)
    torch.testing.assert_allclose(
        postprocessed.batch_sizes, sample_packed_sequence.batch_sizes
    )


def test_seq_rnn(sample_packed_sequence):
    """Test running SeqRNN on sample data."""
    seq_rnn = SeqRNN(
        nn.GRU, input_size=16, hidden_size=16, num_layers=2, batch_first=True
    )

    output = seq_rnn(sample_packed_sequence)

    assert output.data.shape == sample_packed_sequence.data.shape
    torch.testing.assert_allclose(
        output.batch_sizes, sample_packed_sequence.batch_sizes
    )


def test_seq_encoder_dimensions(yolov4_ints, parsed_yolov4):
    """Verify seq encoder dimensions."""
    _, neck, head = parsed_yolov4
    inputs = {}
    for key, value in yolov4_ints.items():
        inputs[key] = nn.utils.rnn.PackedSequence(
            value.expand(2, -1, -1, -1), torch.tensor([1, 1])
        )

    anchors = deepcopy(head.num_anchors)
    out_channels = deepcopy(neck.out_channels)
    seq_encoder = SeqEncoder(
        anchors=anchors,
        out_channels=out_channels,
    )

    output = seq_encoder(inputs)

    for key in yolov4_ints.keys():
        assert output[key].data.shape == (
            2,
            out_channels[key],
            *yolov4_ints[key].shape[2:],
        )
