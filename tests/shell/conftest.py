import pytest


@pytest.fixture(
    params=[
        "${hydra:runtime.cwd}/tests/unit/data/yolov4.cfg",
        "${hydra:runtime.cwd}/tests/unit/data/yolov4-tiny.cfg",
    ]
)
def base_command(request):
    """Return base command for DIR."""
    return ["train.py", f"darknet.config={request.param}"]
