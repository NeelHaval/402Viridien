"""Shared deterministic stress parameters; no private data or network fixtures."""
import os

import pytest


def pytest_addoption(parser):
    group=parser.getgroup('viridian stress testing')
    group.addoption('--stress-seed',type=int,default=int(os.environ.get('VIRIDIAN_STRESS_SEED','20261108')))
    group.addoption('--stress-iterations',type=int,default=int(os.environ.get('VIRIDIAN_STRESS_ITERATIONS','50')))


@pytest.fixture(scope='session')
def stress_seed(request):
    return request.config.getoption('--stress-seed')


@pytest.fixture(scope='session')
def stress_iterations(request):
    value=request.config.getoption('--stress-iterations')
    if not 1<=value<=10000:raise pytest.UsageError('--stress-iterations must be between 1 and 10000')
    return value


def pytest_report_header(config):
    return f"Isolated stress checks: seed={config.getoption('--stress-seed')}, iterations={config.getoption('--stress-iterations')}"
