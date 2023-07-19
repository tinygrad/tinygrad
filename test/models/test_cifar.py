import unittest
from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from tinygrad.state import get_parameters
from examples.hlb_cifar10 import SpeedyResNet
from tinygrad.jit import TinyJit, JIT_SUPPORTED_DEVICE
from tinygrad.ops import GlobalCounters
from tinygrad.lazy import Device

class TestCifar(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT in JIT_SUPPORTED_DEVICE, "needs JIT")
  def test_train_step(self):
    # TODO: with default device
    #old_default = Device.DEFAULT
    #Device.DEFAULT = "FAKE"
    #Device['fake'].codegen = Device[old_default].codegen

    # TODO: with train
    old_training = Tensor.training
    Tensor.training = True

    model = SpeedyResNet()
    optimizer = optim.SGD(get_parameters(model), lr=0.01, momentum=0.8, nesterov=True, weight_decay=0.15)

    @TinyJit
    def train(X):
      out = model(X)
      loss = out.mean()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    for _ in range(3): train(Tensor.randn(32, 3, 32, 32))
    print(f"used {GlobalCounters.mem_used/1e9} GB")
    print(len(train.jit_cache))
    assert GlobalCounters.mem_used/1e9 < 0.55, "CIFAR used more than 0.55 GB"
    assert len(train.jit_cache) <= 236, "CIFAR training than 236 kernels"

    # reset device
    Tensor.training = old_training
    #Device.DEFAULT = old_default

if __name__ == '__main__':
  unittest.main()
