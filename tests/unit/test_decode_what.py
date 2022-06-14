"""Tests for what decoder."""
import pytest
import torch

from src.models.components.decode.what import WhatDecoder


@pytest.mark.parametrize("latent_dim", [2, 4])
@pytest.mark.parametrize("decoded_size", [2, 8])
@pytest.mark.parametrize("n_objects", [3, 7])
def test_what_decoder_dimensions(latent_dim, decoded_size, n_objects):
    """Verify what decoder output dimensions."""
    z_what = torch.rand(n_objects, latent_dim)
    what_dec = WhatDecoder(latent_dim=latent_dim, decoded_size=decoded_size)
    decoded = what_dec(z_what)
    assert decoded.shape == (n_objects, 3, decoded_size, decoded_size)


def test_what_decoder_dtype():
    """Verify what decoder output dtype."""
    latent_dim = 5
    z_what = torch.rand(3, 4, latent_dim)
    what_dec = WhatDecoder(latent_dim=latent_dim)
    decoded = what_dec(z_what)
    assert decoded.dtype == torch.float
    assert (decoded >= 0).all()
    assert (decoded <= 1).all()
