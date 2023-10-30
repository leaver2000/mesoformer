import os

import pytest
import torch

_device = torch.device(os.environ.get("TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


@pytest.fixture
def device() -> torch.device:
    return _device
