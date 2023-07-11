import pytest
from tinygrad.helpers import getenv

@pytest.fixture(autouse=getenv("CUDACPU", 0) and getenv("TRITON", 0))
def mock_torch(monkeypatch):
  monkeypatch.setattr("triton.runtime.jit.get_current_device", lambda: 0)
