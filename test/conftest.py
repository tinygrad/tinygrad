import pytest
from tinygrad.helpers import getenv

@pytest.fixture(autouse=getenv("CUDACPU", 0) and getenv("TRITON", 0))
def mock_torch(monkeypatch):
  monkeypatch.setattr("triton.compiler.compiler.ptx_to_cubin", lambda ptx, arch: lambda src: src)
