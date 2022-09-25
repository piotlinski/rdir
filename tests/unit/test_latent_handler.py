"""Test handling latents in DIR."""
import pytest
import torch

from src.models.components.latents import LatentHandler


def test_reset_latent():
    """Verify if latent values are reset to given value."""
    latent = torch.rand(3)
    mask = torch.tensor([False, True, False])
    value = torch.tensor(999)

    reset = LatentHandler.reset_latent(latent, mask, value)

    assert reset[0] == reset[2] == value
    assert reset[1] == latent[1]


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("n_objects", [4, 5])
@pytest.mark.parametrize("z_what_size", [4, 8])
def test_reset_non_present(batch_size, n_objects, z_what_size):
    latent_handler = LatentHandler()
    latents = (
        torch.rand(batch_size, n_objects, 4),
        torch.zeros(batch_size, n_objects, 1),
        (
            torch.rand(batch_size, n_objects, z_what_size),
            torch.rand(batch_size, n_objects, z_what_size),
        ),
        (
            torch.rand(batch_size, n_objects, 1),
            torch.rand(batch_size, n_objects, 1),
        ),
    )

    (
        _,
        _,
        (reset_z_what_loc, reset_z_what_scale),
        (reset_z_depth_loc, reset_z_depth_scale),
    ) = latent_handler.reset_non_present(latents)

    assert (reset_z_what_loc == latent_handler._empty_loc).all()
    assert (reset_z_what_scale == latent_handler._empty_scale).all()
    assert (reset_z_depth_loc == latent_handler._empty_loc).all()
    assert (reset_z_depth_scale == latent_handler._empty_scale).all()


@pytest.mark.parametrize("n_added, ", [2, 3, 5, 7])
def test_negative_indices(n_added):
    """Verify if negative indices are generated."""
    z_present = torch.tensor(
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ).unsqueeze(-1)

    negative = LatentHandler.negative_indices(z_present, n_added)

    assert negative[0, 0] == -1
    assert negative[0, 1] == -1
    assert negative[1, 0] == -1


@pytest.mark.parametrize("negative_percentage", [0.0, 0.5])
def test_add_negative(negative_percentage):
    """Verify adding negative indices to tensor."""
    latent_handler = LatentHandler(negative_percentage=negative_percentage)
    z_present = torch.tensor(
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ).unsqueeze(-1)

    modified = latent_handler.add_negative(z_present)

    positive = torch.nonzero(z_present, as_tuple=True)
    assert modified.shape == z_present.shape
    assert torch.equal(modified[positive], z_present[positive])
    summed = torch.sum(modified.bool(), dim=1)
    assert summed[0] == summed[1] == summed[2]


@pytest.mark.parametrize("negative_percentage", [0.0, 0.2, 0.5])
def test_filter_representation(negative_percentage):
    """Test if representation is filtered appropriately."""
    latent_handler = LatentHandler(negative_percentage=negative_percentage)
    batch_size = 2
    n_objects = 20
    z_what_size = 3
    z_where = torch.rand(batch_size, n_objects, 4)
    z_present = torch.randint(0, 2, (batch_size, n_objects, 1))
    z_what = torch.rand(batch_size, n_objects, z_what_size)
    z_depth = torch.rand(batch_size, n_objects, 1)

    nz_where, nz_present, nz_what, nz_depth = latent_handler.filter_representation(
        (z_where, z_present, z_what, z_depth)
    )

    n_filtered = nz_present.shape[1]
    assert nz_where.shape == (batch_size, n_filtered, z_where.shape[-1])
    assert nz_present.shape == (batch_size, n_filtered, z_present.shape[-1])
    assert nz_what.shape == (batch_size, n_filtered, z_what_size)
    assert nz_depth.shape == (batch_size, n_filtered, z_depth.shape[-1])
