import pytest

from tests.helpers.run_command import run_command

"""
A couple of tests executing hydra sweeps.

Use the following command to skip slow tests:
    pytest -k "not slow"
"""


@pytest.mark.slow
def test_experiments(base_command):
    """Test running all available experiment configs for 1 epoch."""
    command = [*base_command, "-m", "experiment=glob(*)", "++trainer.max_epochs=1"]
    run_command(command)


@pytest.mark.slow
def test_default_sweep(base_command):
    """Test default Hydra sweeper."""
    command = [
        *base_command,
        "-m",
        "datamodule.batch_size=64,128",
        "model.lr=0.01,0.02",
        "trainer=default",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
def test_optuna_sweep(base_command):
    """Test Optuna sweeper."""
    command = [
        *base_command,
        "-m",
        "hparams_search=mnist_optuna",
        "trainer=default",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)
