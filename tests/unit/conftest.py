from unittest.mock import mock_open, patch

import pytest
import torch
from yolov4 import YOLOv4

from src.models.components.decode.decoder import Decoder
from src.models.components.encode.encoder import Encoder
from src.models.components.encode.parse import parse_yolov4


@pytest.fixture(scope="session")
def yolov4_cfgs():
    """Sample YOLOv4 architecture configs."""
    with open("tests/unit/data/yolov4.cfg") as fp:
        yolov4_cfg = fp.read()
    with open("tests/unit/data/yolov4-tiny.cfg") as fp:
        yolov4_tiny_cfg = fp.read()
    return {"yolov4": yolov4_cfg, "yolov4-tiny": yolov4_tiny_cfg}


@pytest.fixture(scope="session", params=["yolov4", "yolov4-tiny"])
def yolov4_cfg(request, yolov4_cfgs):
    """Parametrized sample YOLOv4 architecture config."""
    return yolov4_cfgs[request.param]


@pytest.fixture(scope="session")
def yolov4(yolov4_cfg):
    """Sample YOLOv4 architecture."""
    with patch("builtins.open", mock_open(read_data=yolov4_cfg)):
        return YOLOv4("test")


@pytest.fixture(scope="session")
def parsed_yolov4(yolov4_cfg):
    """Sample parsed YOLOv4 model."""
    with patch("builtins.open", mock_open(read_data=yolov4_cfg)):
        return parse_yolov4("test")


@pytest.fixture(scope="session", params=[(7, 160, 2), (42, 320, 1)])
def sample_inputs(request):
    """Return sample inputs for YOLOv4."""
    seed, image_size, batch_size = request.param
    generator = torch.random.manual_seed(seed)
    return torch.rand(batch_size, 3, image_size, image_size, generator=generator)


@pytest.fixture(scope="session")
def yolov4_feats(parsed_yolov4, sample_inputs):
    """Return sample output of YOLOv4 backbone."""
    backbone, *_ = parsed_yolov4
    return backbone(sample_inputs)


@pytest.fixture(scope="session")
def yolov4_ints(parsed_yolov4, yolov4_feats):
    """Return sample output of YOLOv4 neck."""
    _, neck, _ = parsed_yolov4
    return neck(yolov4_feats)


@pytest.fixture(scope="session")
def yolov4_preds(parsed_yolov4, yolov4_ints):
    """Return sample output of YOLOv4 head."""
    *_, head = parsed_yolov4
    return head(yolov4_ints)


@pytest.fixture(scope="module")
def yolov4_mock(parsed_yolov4):
    """Mock parsing yolov4."""
    with patch(
        "src.models.components.encode.encoder.parse_yolov4", return_value=parsed_yolov4
    ) as _mock:
        yield _mock


@pytest.fixture
@patch("src.models.components.encode.encoder.parse_yolov4")
def encoder(parse_yolov4_mock, parsed_yolov4) -> Encoder:
    """Encoder for testing DIR."""
    parse_yolov4_mock.return_value = parsed_yolov4
    return Encoder(yolo=("test", None), z_what_size=4)


@pytest.fixture
def decoder() -> Decoder:
    """Decoder for testing DIR."""
    return Decoder(z_what_size=4, image_size=192)
