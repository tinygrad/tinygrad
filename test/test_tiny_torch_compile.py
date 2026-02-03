#!/usr/bin/env python
import unittest
import torch

import tinygrad.nn.torch  # noqa: F401


def has_tiny_backend() -> bool:
  try:
    return "tiny" in torch._dynamo.list_backends()
  except Exception:
    return False


@unittest.skipUnless(hasattr(torch, "compile") and has_tiny_backend(), "tiny torch.compile backend unavailable")
class TestTinyTorchCompile(unittest.TestCase):
  def test_backend_registered(self):
    self.assertIn("tiny", torch._dynamo.list_backends())

  def test_compile_creates_callable(self):
    def foo(x): return x * 2
    compiled = torch.compile(foo, backend="tiny")
    self.assertTrue(callable(compiled))


if __name__ == "__main__":
  unittest.main()
#!/usr/bin/env python
import unittest
import torch

import tinygrad.nn.torch  # noqa: F401


def has_tiny_backend() -> bool:
  try:
    return "tiny" in torch._dynamo.list_backends()
  except Exception:
    return False


@unittest.skipUnless(hasattr(torch, "compile") and has_tiny_backend(), "tiny torch.compile backend unavailable")
class TestTinyTorchCompile(unittest.TestCase):
  def test_backend_registered(self):
    self.assertIn("tiny", torch._dynamo.list_backends())

  def test_compile_creates_callable(self):
    def foo(x): return x * 2
    compiled = torch.compile(foo, backend="tiny")
    self.assertTrue(callable(compiled))


if __name__ == "__main__":
  unittest.main()

