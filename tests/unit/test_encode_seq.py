"""Tests for sequential encoder."""
import pytest
import torch
import torch.nn as nn

from src.models.components.encode.seq import SeqEncoder


@pytest.mark.parametrize("rnn_cls", [nn.GRU, nn.LSTM])
@pytest.mark.parametrize("depth_probabilistic", [False, True])
@pytest.mark.parametrize("what_probabilistic", [False, True])
def test_seq_encoder_layers(what_probabilistic, depth_probabilistic, rnn_cls):
    """Test if encoders are instantiated according to parameters."""
    seq_enc = SeqEncoder(
        n_objects=2,
        z_what_size=4,
        what_probabilistic=what_probabilistic,
        depth_probabilistic=depth_probabilistic,
        rnn_cls=rnn_cls,
        num_layers=1,
        bidirectional=False,
    )

    assert seq_enc.what_rnn.__class__ == rnn_cls
    assert seq_enc.depth_rnn.__class__ == rnn_cls
    assert (seq_enc.what_scale_rnn.__class__ == rnn_cls) is what_probabilistic
    assert (seq_enc.depth_scale_rnn.__class__ == rnn_cls) is depth_probabilistic


@pytest.mark.parametrize("depth_probabilistic", [False, True])
@pytest.mark.parametrize("what_probabilistic", [False, True])
def test_seq_encoder_alters(what_probabilistic, depth_probabilistic):
    """Test if sequential encoder alters latents."""
    batch_size = 2
    seq_len = 5
    n_objects = 4
    z_what_size = 3
    z_where = torch.rand(batch_size, seq_len, n_objects, 4)
    z_present = torch.randint(0, 2, (batch_size, seq_len, n_objects, 1))
    z_what = torch.rand(batch_size, seq_len, n_objects, z_what_size)
    z_what_scale = torch.rand(batch_size, seq_len, n_objects, z_what_size)
    z_depth = torch.rand(batch_size, seq_len, n_objects, 1)
    z_depth_scale = torch.rand(batch_size, seq_len, n_objects, 1)
    seq_enc = SeqEncoder(
        n_objects=n_objects,
        z_what_size=z_what_size,
        what_probabilistic=what_probabilistic,
        depth_probabilistic=depth_probabilistic,
    )

    (
        new_z_where,
        new_z_present,
        (new_z_what, new_z_what_scale),
        (new_z_depth, new_z_depth_scale),
    ) = seq_enc((z_where, z_present, (z_what, z_what_scale), (z_depth, z_depth_scale)))

    assert new_z_where.shape == z_where.shape
    assert torch.allclose(new_z_where, z_where)
    assert new_z_present.shape == z_present.shape
    assert torch.allclose(new_z_present, z_present)
    assert new_z_what.shape == z_what.shape
    assert not torch.allclose(new_z_what, z_what)
    assert new_z_what_scale.shape == z_what_scale.shape
    assert torch.allclose(new_z_what_scale, z_what_scale) is not what_probabilistic
    assert new_z_depth.shape == z_depth.shape
    assert not torch.allclose(new_z_depth, z_depth)
    assert new_z_depth_scale.shape == z_depth_scale.shape
    assert torch.allclose(new_z_depth_scale, z_depth_scale) is not depth_probabilistic
