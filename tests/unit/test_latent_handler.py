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
