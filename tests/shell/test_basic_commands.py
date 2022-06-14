import pytest

from tests.helpers.run_command import run_command
from tests.helpers.runif import RunIf

"""
A couple of sanity checks to make sure the model doesn't crash with different running options.
"""


@pytest.mark.slow
def test_fast_dev_run(base_command):
    """Test running for 1 train, val and test batch."""
    command = [*base_command, "++trainer.fast_dev_run=true"]
    run_command(command)


@pytest.mark.slow
def test_cpu(base_command):
    """Test running 1 epoch on CPU."""
    command = [*base_command, "++trainer.max_epochs=1", "++trainer.gpus=0"]
    run_command(command)


# use RunIf to skip execution of some tests, e.g. when no gpus are available
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_gpu(base_command):
    """Test running 1 epoch on GPU."""
    command = [
        *base_command,
        "++trainer.max_epochs=1",
        "++trainer.gpus=1",
    ]
    run_command(command)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_mixed_precision(base_command):
    """Test running 1 epoch with pytorch native automatic mixed precision (AMP)."""
    command = [
        *base_command,
        "++trainer.max_epochs=1",
        "++trainer.gpus=1",
        "++trainer.precision=16",
    ]
    run_command(command)


@pytest.mark.slow
def test_double_validation_loop(base_command):
    """Test running 1 epoch with validation loop twice per epoch."""
    command = [
        *base_command,
        "++trainer.max_epochs=1",
        "++trainer.val_check_interval=0.5",
    ]
    run_command(command)
