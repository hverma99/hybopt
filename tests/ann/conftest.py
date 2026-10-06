import pytest
import torch


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)  # as in scripts/train.py, so training results are reproducible
