"""Tests for DIR decoder."""
import pytest
import torch

from src.models.components.decode.decoder import Decoder


@pytest.mark.parametrize("batch_size", [4, 3, 7])
@pytest.mark.parametrize("n_objects", [5, 7, 10])
@pytest.mark.parametrize("no_objects", [[0], [1, 2], []])
def test_fix_z_present(batch_size, n_objects, no_objects):
    """Test if z_present with no objects in images is fixed appropriately."""
    z_present = torch.randint(2, (batch_size, n_objects, 1))
    z_present[:, 0] = 1
    z_present[no_objects] = 0
    n_present = torch.sum(z_present, dim=1)
    max_objects = torch.max(n_present)

    fixed = Decoder.fix_z_present(z_present)

    assert (torch.sum(fixed, dim=1) > 0).all()
    fixed_n_present = torch.sum(fixed, dim=1)
    assert torch.max(fixed_n_present) == max_objects
    assert (fixed_n_present[no_objects] == max_objects).all()


def test_filter_representation():
    """Test if representation is filtered appropriately."""
    batch_size = 2
    n_objects = 4
    z_what_size = 3
    z_where = torch.rand(batch_size, n_objects, 4)
    z_present = torch.randint(0, 2, (batch_size, n_objects, 1))
    z_what = torch.rand(batch_size, n_objects, z_what_size)
    z_depth = torch.rand(batch_size, n_objects, 1)

    decoder = Decoder(z_what_size=z_what_size)
    new_z_where, new_z_present, new_z_what, new_z_depth = decoder.filter_representation(
        (z_where, z_present, z_what, z_depth)
    )

    n_filtered = torch.sum(z_present, dtype=torch.long)
    assert new_z_where.shape == (n_filtered, z_where.shape[-1])
    assert new_z_what.shape == (n_filtered, z_what_size)
    assert new_z_depth.shape == (n_filtered, z_depth.shape[-1])


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("n_objects", [1, 2])
@pytest.mark.parametrize("z_what_size", [2, 5])
@pytest.mark.parametrize("decoded_size", [4, 16])
def test_decode_objects(batch_size, n_objects, z_what_size, decoded_size):
    """Verify dimension of decoded objects."""
    decoder = Decoder(z_what_size=z_what_size, decoded_size=decoded_size)
    z_what = torch.rand(batch_size, n_objects, z_what_size)
    z_where = torch.rand(batch_size, n_objects, 4)
    decoded_images, z_where_flat = decoder.decode_objects(z_where, z_what)
    assert decoded_images.shape == (
        batch_size * n_objects,
        3,
        decoded_size,
        decoded_size,
    )
    assert z_where_flat.shape == (batch_size * n_objects, 4)


@pytest.mark.parametrize(
    "n_present, expected",
    [
        (torch.tensor([1, 3, 2]), torch.tensor([0, 1, 0, 0, 0, 2, 3, 4, 0, 5, 6, 0])),
        (torch.tensor([1, 2, 2]), torch.tensor([0, 1, 0, 0, 2, 3, 0, 4, 5])),
        (torch.tensor([3, 1, 1]), torch.tensor([0, 1, 2, 3, 0, 4, 0, 0, 0, 5, 0, 0])),
    ],
)
def test_pad_indices(n_present, expected):
    """Verify padded indices calculation."""
    indices = Decoder.pad_indices(n_present)
    assert indices.shape == (n_present.shape[0] * (torch.max(n_present) + 1),)
    assert torch.max(indices) == torch.sum(n_present)
    assert torch.equal(indices, expected)


def test_pad_reconstructions():
    """Verify padding reconstructions in Decoder."""
    image_size = 320
    decoder = Decoder(image_size=image_size)
    objects = (
        torch.arange(1, 5, dtype=torch.float)
        .view(-1, 1, 1, 1)
        .expand(4, 3, image_size, image_size)
    )
    z_depth = torch.arange(5, 9, dtype=torch.float).view(-1, 1)
    n_present = torch.tensor([1, 3])

    padded_objects, padded_z_depth = decoder.pad_reconstructions(
        transformed_objects=objects, z_depth=z_depth, n_present=n_present
    )

    assert padded_objects.shape == (2, 4, 3, image_size, image_size)
    assert padded_z_depth.shape == (2, 4, 1)
    assert torch.equal(padded_objects[0][1], objects[0])
    assert torch.equal(padded_objects[0][2], padded_objects[0][3])
    assert torch.equal(padded_objects[0][0], padded_objects[1][0])
    assert torch.equal(padded_objects[1][1], objects[1])
    assert torch.equal(padded_objects[1][2], objects[2])
    assert torch.equal(padded_objects[1][3], objects[3])
    assert padded_z_depth[0][1] == z_depth[0]
    assert padded_z_depth[0][2] == padded_z_depth[0][3] == decoder.EMPTY_DEPTH
    assert padded_z_depth[0][0] == padded_z_depth[1][0]
    assert padded_z_depth[1][1] == z_depth[1]
    assert padded_z_depth[1][2] == z_depth[2]
    assert padded_z_depth[1][3] == z_depth[3]


@pytest.mark.parametrize("decoded_size", [4, 16])
@pytest.mark.parametrize("image_size", [128, 196])
def test_transform_objects(decoded_size, image_size):
    """Verify dimension of transformed objects."""
    decoder = Decoder(decoded_size=decoded_size, image_size=image_size)
    decoded_objects = torch.rand(6, 3, decoded_size, decoded_size)
    z_where_flat = torch.rand(6, 4)
    z_present = torch.tensor([[[1], [0], [1]], [[0], [1], [1]]], dtype=torch.long)
    z_depth = torch.rand(6, 1)

    objects, depths = decoder.transform_objects(
        decoded_objects, z_where_flat, z_present, z_depth
    )

    assert objects.shape[2:] == (3, image_size, image_size)
    assert depths.shape[0] == objects.shape[0]
    assert depths.shape[1] == objects.shape[1]


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


@pytest.mark.parametrize("train_what", [False, True])
def test_decoder_train(train_what):
    """Verify if training submodules can be disabled."""
    decoder = Decoder(train_what=train_what)
    assert all(
        param.requires_grad == train_what for param in decoder.what_dec.parameters()
    )
