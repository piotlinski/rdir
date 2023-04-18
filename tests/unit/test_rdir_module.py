"""Tests for R-DIR model."""
from functools import partial

import pyro
import pytest
import torch

from src.models.rdir_module import RDIR
from src.models.dir_module import DIR


@pytest.fixture
def dir(yolov4_mock, encoder, decoder):
    """Return DIR model for R-DIR tests."""
    return DIR(encoder, decoder)


def test_rdir_encoder_forward(dir):
    """Verify DIR encoder_forward output dimensions and dtypes."""
    batch_size = 2
    n_objects = 3
    image_size = 192
    model = RDIR(dir)
    inputs = torch.rand(batch_size, n_objects, 3, image_size, image_size)
    packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
        inputs, [n_objects] * batch_size, batch_first=True
    )

    latents = model.encoder_forward(packed_inputs)

    z_where, z_present, (z_what, z_what_scale), (z_depth, z_what_scale) = latents
    grid_objects = z_where.data.shape[1]
    assert z_where.data.shape == (batch_size * n_objects, grid_objects, 4)
    assert z_where.data.dtype == torch.float
    assert z_present.data.shape == (batch_size * n_objects, grid_objects, 1)
    assert z_present.data.dtype == torch.float
    assert (z_present.data >= 0).all()
    assert (z_present.data <= 1).all()
    assert z_what.data.shape == (batch_size * n_objects, grid_objects, 4)
    assert z_what.data.dtype == torch.float
    assert z_depth.data.shape == (batch_size * n_objects, grid_objects, 1)
    assert z_depth.data.dtype == torch.float


def test_rdir_decoder_forward(dir):
    """Verify running decoder_forward in RDIR."""
    batch_size = 2
    n_objects = 3
    grid_objects = 9
    z_what_size = 4
    image_size = 192
    model = RDIR(dir)
    pack = partial(
        torch.nn.utils.rnn.pack_padded_sequence,
        lengths=[n_objects] * batch_size,
        batch_first=True,
    )
    z_where = pack(torch.rand(batch_size, n_objects, grid_objects, 4))
    z_present = pack(torch.randint(0, 2, (batch_size, n_objects, grid_objects, 1)))
    z_what = pack(torch.rand(batch_size, n_objects, grid_objects, z_what_size))
    z_depth = pack(torch.rand(batch_size, n_objects, grid_objects, 1))
    latents = (z_where, z_present, z_what, z_depth)

    reconstructions = model.decoder_forward(latents)

    assert reconstructions.data.shape == (
        batch_size * n_objects,
        3,
        image_size,
        image_size,
    )
    assert (reconstructions.data >= 0).all()
    assert (reconstructions.data <= 1).all()


def test_rdir_model_guide(dir):
    """Test running model and guide in R-DIR."""
    batch_size = 2
    n_objects = 3
    image_size = 192
    model = RDIR(dir)
    inputs = torch.rand(batch_size, n_objects, 3, image_size, image_size)
    packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
        inputs, [n_objects] * batch_size, batch_first=True, enforce_sorted=False
    )
    criterion = pyro.infer.Trace_ELBO().differentiable_loss

    loss = criterion(model.model, model.guide, packed_inputs)

    assert not loss.isnan()
