import pytest

from tests.helpers.run_command import run_command


@pytest.mark.slow
def test_debug_default(base_command):
    command = [*base_command, "debug=default"]
    run_command(command)


@pytest.mark.slow
def test_debug_limit_batches(base_command):
    command = [*base_command, "debug=limit_batches"]
    run_command(command)


@pytest.mark.slow
def test_debug_overfit(base_command):
    command = [*base_command, "debug=overfit"]
    run_command(command)


@pytest.mark.slow
def test_debug_profiler(base_command):
    command = [*base_command, "debug=profiler"]
    run_command(command)


@pytest.mark.slow
def test_debug_step(base_command):
    command = [*base_command, "debug=step"]
    run_command(command)
