import unittest
import torch
from torch import nn, optim

import tinygrad.nn.torch  # noqa: F401  # pylint: disable=unused-import


class TestTorchCompileRegression(unittest.TestCase):
  @unittest.skipIf(not hasattr(torch, "compile"), "torch.compile is unavailable")
  def test_compile_tiny_backend_runs(self):
    def foo(x, y):
      return torch.sin(x) + torch.cos(y)

    opt_foo = torch.compile(foo, backend="tiny")
    for _ in range(2):
      out = opt_foo(torch.randn(10, 10), torch.randn(10, 10))
    self.assertEqual(out.shape, (10, 10))
    self.assertEqual(out.device.type, "tiny")

  @unittest.skipIf(not hasattr(torch, "compile"), "torch.compile is unavailable")
  def test_compile_tiny_backend_tuple_outputs(self):
    def foo(x, y):
      a = torch.sin(x)
      b = torch.cos(y)
      return a, b, a + b

    opt_foo = torch.compile(foo, backend="tiny")
    for _ in range(2):
      outs = opt_foo(torch.randn(10, 10), torch.randn(10, 10))

    self.assertEqual(len(outs), 3)
    for out in outs:
      self.assertEqual(out.shape, (10, 10))
      self.assertEqual(out.device.type, "tiny")

  @unittest.skipIf(not hasattr(torch, "compile"), "torch.compile is unavailable")
  def test_compile_tiny_backend_training_step(self):
    model = nn.Linear(16, 4).cpu()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    def step(x, y):
      out = model(x)
      loss = loss_fn(out, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      return loss

    compiled_step = torch.compile(step, backend="tiny")
    for _ in range(2):
      loss = compiled_step(torch.randn(8, 16), torch.randn(8, 4))
    self.assertTrue(torch.isfinite(loss).item())

  @unittest.skipIf(not hasattr(torch, "compile"), "torch.compile is unavailable")
  def test_compile_tiny_backend_runs_with_tiny_inputs(self):
    def foo(x, y):
      return torch.sin(x) + torch.cos(y)

    opt_foo = torch.compile(foo, backend="tiny")
    for _ in range(2):
      out = opt_foo(torch.randn(10, 10, device="tiny"), torch.randn(10, 10, device="tiny"))
    self.assertEqual(out.shape, (10, 10))
    self.assertEqual(out.device.type, "tiny")

  @unittest.skipIf(not hasattr(torch, "compile"), "torch.compile is unavailable")
  def test_compile_tiny_backend_training_step_all_tiny(self):
    model = nn.Linear(16, 4).to("tiny")
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    def step(x, y):
      out = model(x)
      loss = loss_fn(out, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      return loss

    compiled_step = torch.compile(step, backend="tiny")
    for _ in range(2):
      loss = compiled_step(torch.randn(8, 16, device="tiny"), torch.randn(8, 4, device="tiny"))
    self.assertEqual(loss.device.type, "tiny")
    self.assertTrue(torch.isfinite(loss).item())

  @unittest.skipIf(not hasattr(torch, "compile"), "torch.compile is unavailable")
  def test_compile_tiny_backend_training_step_batchnorm_all_tiny(self):
    model = nn.Sequential(
      nn.Linear(16, 16),
      nn.BatchNorm1d(16),
      nn.ReLU(),
      nn.Linear(16, 4),
    ).to("tiny")
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    def step(x, y):
      out = model(x)
      loss = loss_fn(out, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      return loss

    compiled_step = torch.compile(step, backend="tiny")
    for _ in range(2):
      loss = compiled_step(torch.randn(8, 16, device="tiny"), torch.randn(8, 4, device="tiny"))
    self.assertEqual(loss.device.type, "tiny")
    self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
  unittest.main()
