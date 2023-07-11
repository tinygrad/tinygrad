import pytest
from tinygrad.helpers import getenv

@pytest.fixture(autouse=getenv("CUDACPU", 0) and getenv("TRITON", 0))
def mock_torch(monkeypatch):
  print("SANITY CHECK: MOCKING TRITON")
  monkeypatch.setattr("triton.compiler.compiler.get_architecture_descriptor", lambda _: 86)
