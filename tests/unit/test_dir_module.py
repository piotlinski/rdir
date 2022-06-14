"""Tests for DIR model."""
from unittest.mock import patch

import pyro
import pytest
import torch

from src.models.components.decode.decoder import Decoder
from src.models.components.encode.encoder import Encoder
from src.models.dir_module import DIR


@pytest.fixture
@patch("src.models.components.encode.encoder.parse_yolov4")
def encoder(parse_yolov4_mock, parsed_yolov4) -> Encoder:
    """Encoder for testing DIR."""
    parse_yolov4_mock.return_value = parsed_yolov4
    return Encoder(yolo_cfg_file="test", z_what_size=4)


@pytest.fixture
def decoder() -> Decoder:
    """Decoder for testing DIR."""
    return Decoder(z_what_size=4, image_size=192)


@pytest.mark.parametrize(
    "z_what_scale_const, z_depth_scale_const, z_present_threshold, is_deterministic",
    [
        (1, 0, -1, False),
        (0, 2, -1, False),
        (0, 0, -1, False),
        (0, 0, 0.2, True),
        (0, 0, 0.5, True),
        (0, 2, 0.3, False),
        (0.2, 0, 0.7, False),
    ],
)
def test_dir_is_deterministic(
    z_what_scale_const,
    z_depth_scale_const,
    z_present_threshold,
    is_deterministic,
    decoder,
    yolov4_mock,
):
    """Check if deterministic flag is set correctly."""
    model = DIR(
        encoder=Encoder(
            yolo_cfg_file="test",
            z_what_scale_const=z_what_scale_const,
            z_depth_scale_const=z_depth_scale_const,
        ),
        decoder=decoder,
        z_present_threshold=z_present_threshold,
    )
    assert model.is_deterministic == is_deterministic


@pytest.mark.parametrize("z_what_scale_const", [-1.0, 0, 0.1])
@pytest.mark.parametrize("z_depth_scale_const", [-1.0, 0, 0.2])
@pytest.mark.parametrize("z_present_threshold", [-1.0, 0.1])
@pytest.mark.parametrize("reset_non_present", [False, True])
@pytest.mark.parametrize("clone_backbone", [False, True])
def test_dir_encoder_forward(
    z_what_scale_const,
    z_depth_scale_const,
    z_present_threshold,
    reset_non_present,
    clone_backbone,
    yolov4_mock,
    encoder,
    decoder,
):
    """Verify DIR encoder_forward output dimensions and dtypes."""
    batch_size = 2
    image_size = 160
    model = DIR(
        encoder=encoder,
        decoder=decoder,
        z_present_threshold=z_present_threshold,
    )
    inputs = torch.rand(batch_size, 3, image_size, image_size)

    latents = model.encoder_forward(inputs)

    z_where, z_present, z_what, z_depth = latents
    n_objects = z_where.shape[1]
    assert z_where.shape == (batch_size, n_objects, 4)
    assert z_where.dtype == torch.float
    assert z_present.shape == (batch_size, n_objects, 1)
    assert z_present.dtype == torch.float
    assert (z_present >= 0).all()
    assert (z_present <= 1).all()
    assert z_what.shape == (batch_size, n_objects, 4)
    assert z_what.dtype == torch.float
    assert z_depth.shape == (batch_size, n_objects, 1)
    assert z_depth.dtype == torch.float


@pytest.mark.parametrize("z_present_threshold", [-1.0, 0.3])
@pytest.mark.parametrize("normalize_reconstructions", [False, True])
def test_dir_decoder_forward(
    z_present_threshold, encoder, decoder, normalize_reconstructions
):
    """Verify DIR decoder_forward output dimensions and dtypes."""
    batch_size = 2
    n_objects = 3
    z_what_size = 4
    image_size = 192
    z_where = torch.rand(batch_size, n_objects, 4)
    z_present = torch.randint(0, 2, (batch_size, n_objects, 1))
    z_what = torch.rand(batch_size, n_objects, z_what_size)
    z_depth = torch.rand(batch_size, n_objects, 1)
    latents = (z_where, z_present, z_what, z_depth)
    model = DIR(
        encoder=encoder,
        decoder=decoder,
        z_present_threshold=z_present_threshold,
        normalize_reconstructions=normalize_reconstructions,
    )
    reconstructions = model.decoder_forward(latents)
    assert reconstructions.shape == (batch_size, 3, image_size, image_size)
    assert (reconstructions >= 0).all()
    assert (reconstructions <= 1).all()


@pytest.mark.parametrize(
    "z_present_threshold, z_present_p_prior", [(-1.0, 0.5), (0.3, 0.5)]
)
@pytest.mark.parametrize("normalize_reconstructions", [False, True])
@pytest.mark.parametrize("objects_coef", [0.0, 1.0])
def test_dir_model_guide(
    z_present_threshold,
    z_present_p_prior,
    normalize_reconstructions,
    objects_coef,
    encoder,
    decoder,
    yolov4_mock,
):
    """Verify if DIR model and guide are configured correctly."""
    batch_size = 3
    image_size = 192
    inputs = torch.rand(batch_size, 3, image_size, image_size)
    model = DIR(
        encoder=encoder,
        decoder=decoder,
        z_present_threshold=z_present_threshold,
        z_present_p_prior=z_present_p_prior,
        objects_coef=objects_coef,
        normalize_reconstructions=normalize_reconstructions,
    )
    criterion = pyro.infer.Trace_ELBO().differentiable_loss
    loss = criterion(model.model, model.guide, inputs)
    assert not loss.isnan()
