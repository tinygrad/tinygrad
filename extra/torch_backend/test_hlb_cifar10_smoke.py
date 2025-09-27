import unittest
import torch

# enable the tiny torch backend
import tinygrad.frontend.torch  # noqa: F401
torch.set_default_device("tiny")


class SmallSpeedy(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.c0 = torch.nn.Conv2d(3, 32, 3, padding=1, bias=False)
    self.bn0 = torch.nn.BatchNorm2d(32, eps=1e-12, momentum=0.85, track_running_stats=False)
    self.g1c1 = torch.nn.Conv2d(32, 64, 3, padding=1, bias=False)
    self.g1bn1 = torch.nn.BatchNorm2d(64, eps=1e-12, momentum=0.85, track_running_stats=False)
    self.g1c2 = torch.nn.Conv2d(64, 64, 3, padding=1, bias=False)
    self.g1bn2 = torch.nn.BatchNorm2d(64, eps=1e-12, momentum=0.85, track_running_stats=False)
    # projection to match residual channels for addition (32 -> 64)
    self.proj = torch.nn.Conv2d(32, 64, kernel_size=1, bias=False)
    self.fc = torch.nn.Linear(64, 10, bias=False)
    # freeze BN weight
    self.bn0.weight.requires_grad_(False)
    self.g1bn1.weight.requires_grad_(False)
    self.g1bn2.weight.requires_grad_(False)

  def forward(self, x):
    x = self.c0(x)
    x = torch.nn.functional.max_pool2d(x, 2)
    x = self.bn0(x.float())
    x = torch.nn.functional.gelu(x, approximate="tanh")
    r = x
    x = self.g1bn1(self.g1c1(x).float())
    x = torch.nn.functional.gelu(x, approximate="tanh")
    x = self.g1bn2(self.g1c2(x).float())
    x = torch.nn.functional.gelu(x, approximate="tanh")
    x = x + self.proj(r)
    x = torch.nn.functional.max_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1]))
    x = torch.flatten(x, 1)
    return self.fc(x)


class TestHLBCifarTorchBackend(unittest.TestCase):
  def test_forward_backward_step(self):
    torch.manual_seed(0)
    model = SmallSpeedy().to("tiny")
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    X = torch.randn(32, 3, 32, 32, device="tiny")
    Y = torch.randint(0, 10, (32,), device="tiny")
    out = model(X)
    self.assertEqual(out.shape, (32, 10))
    loss = torch.nn.functional.cross_entropy(out, Y)
    self.assertTrue(torch.isfinite(loss).item())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


if __name__ == "__main__":
  unittest.main()
