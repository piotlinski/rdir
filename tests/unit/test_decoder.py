"""Tests for DIR decoder."""
import pytest
import torch

from src.models.components.decode.decoder import Decoder


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("n_objects", [1, 2])
@pytest.mark.parametrize("z_what_size", [2, 5])
@pytest.mark.parametrize("decoded_size", [4, 16])
def test_decode_objects(batch_size, n_objects, z_what_size, decoded_size):
    """Verify dimension of decoded objects."""
    decoder = Decoder(z_what_size=z_what_size, decoded_size=decoded_size)
    z_what = torch.rand(batch_size, n_objects, z_what_size)
    decoded_images = decoder.decode_objects(z_what)
    assert decoded_images.shape == (
        batch_size * n_objects,
        3,
        decoded_size,
        decoded_size,
    )


@pytest.mark.parametrize("decoded_size", [4, 16])
@pytest.mark.parametrize("image_size", [128, 196])
def test_transform_objects(decoded_size, image_size):
    """Verify dimension of transformed objects."""
    decoder = Decoder(decoded_size=decoded_size, image_size=image_size)
    decoded_objects = torch.rand(6, 3, decoded_size, decoded_size)
    z_where = torch.rand(2, 3, 4)

    objects = decoder.transform_objects(decoded_objects, z_where)

    assert objects.shape == (2, 3, 3, image_size, image_size)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("n_objects", [2, 5])
@pytest.mark.parametrize("image_size", [64, 128])
def test_reconstruct(batch_size, n_objects, image_size):
    """Check if reconstructed images have appropriate shape."""
    objects = torch.rand(batch_size, n_objects, 3, image_size, image_size)
    weights = torch.rand(batch_size, n_objects, 1)
    merged = Decoder.reconstruct(objects, weights)
    assert merged.shape == (batch_size, 3, image_size, image_size)


def test_decoder_output():
    """Verify decoder output dimensions."""
    batch_size = 3
    n_objects = 7
    z_what_size = 8
    decoded_size = 32
    image_size = 96
    z_where = torch.rand(batch_size, n_objects, 4)
    z_present = torch.randint(0, 2, (batch_size, n_objects, 1))
    z_what = torch.rand(batch_size, n_objects, z_what_size)
    z_depth = torch.rand(batch_size, n_objects, 1)
    latents = (z_where, z_present, z_what, z_depth)

    decoder = Decoder(
        z_what_size=z_what_size, decoded_size=decoded_size, image_size=image_size
    )

    outputs = decoder(latents)
    reconstructions = outputs["reconstructions"]
    assert reconstructions.shape == (batch_size, 3, image_size, image_size)
    assert reconstructions.dtype == torch.float
    assert (reconstructions >= 0).all()
    assert (reconstructions <= 1).all()


def test_decoder_output_eval():
    """Verify decoder output dimensions."""
    batch_size = 3
    n_objects = 7
    z_what_size = 8
    decoded_size = 32
    image_size = 96
    z_where = torch.rand(batch_size, n_objects, 4)
    z_present = torch.randint(0, 2, (batch_size, n_objects, 1))
    z_what = torch.rand(batch_size, n_objects, z_what_size)
    z_depth = torch.rand(batch_size, n_objects, 1)
    latents = (z_where, z_present, z_what, z_depth)

    decoder = Decoder(
        z_what_size=z_what_size, decoded_size=decoded_size, image_size=image_size
    ).eval()

    outputs = decoder(latents)
    reconstructions = outputs["reconstructions"]
    assert reconstructions.shape == (batch_size, 3, image_size, image_size)
    assert reconstructions.dtype == torch.float
    assert (reconstructions >= 0).all()
    assert (reconstructions <= 1).all()


@pytest.mark.parametrize("train_what", [False, True])
def test_decoder_train(train_what):
    """Verify if training submodules can be disabled."""
    decoder = Decoder(train_what=train_what)
    assert all(
        param.requires_grad == train_what for param in decoder.what_dec.parameters()
    )
