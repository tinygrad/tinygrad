# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
import unittest
import torch
import torch._dynamo
import extra.torch_backend.backend  # noqa: F401 # registers the "tiny" backend

class TestTorchCompile(unittest.TestCase):
  def test_torch_compile_basic(self):
    """Test that torch.compile with the tiny backend works."""
    def foo(x, y):
      a = torch.sin(x)
      b = torch.cos(y)
      return a + b

    opt_foo = torch.compile(foo, backend="tiny")
    # run multiple times to test caching
    for _ in range(5):
      out = opt_foo(torch.randn(10, 10), torch.randn(10, 10))
      # outputs are on CPU when inputs are CPU (for backward compatibility)
      self.assertEqual(str(out.device), "cpu")

  def test_torch_compile_backward(self):
    """Test that backward pass works with torch.compile tiny backend."""
    @torch.compile(backend="tiny")
    def matmul(x, w):
      return x @ w

    x = torch.randn(4, 8, requires_grad=True)
    w = torch.randn(8, 4, requires_grad=True)

    out = matmul(x, w)
    loss = out.sum()
    loss.backward()

    self.assertEqual(x.grad.shape, x.shape)
    self.assertEqual(w.grad.shape, w.shape)

  def test_torch_compile_model(self):
    """Test that a compiled model can train."""
    class SimpleModel(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 10)
      def forward(self, x):
        return self.fc(x)

    model = torch.compile(SimpleModel(), backend="tiny")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # training step
    x = torch.randn(4, 16)
    y = torch.randint(0, 10, (4,))
    out = model(x)
    loss = loss_fn(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    self.assertIsInstance(loss.item(), float)

if __name__ == "__main__":
  unittest.main()
