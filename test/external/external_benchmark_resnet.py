import functools
import time
import unittest

from tinygrad import Tensor, TinyJit
from tinygrad.helpers import getenv, GlobalCounters
from tinygrad import nn
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters

from extra.models import resnet
from examples.mlperf.initializers import Conv2dHeNormal, Linear
from examples.hlb_cifar10 import UnsyncedBatchNorm

bs = getenv("BS", 64)

class BenchmarkResnetTrain(unittest.TestCase):
  def _get_layer(self, layer_i, slice_i):
    # isolate to conv, with or without BN
    conv = getenv("CONV", 0)
    bn = getenv("BN", 0)

    if not hasattr(self, 'model'):
      resnet.Conv2d = Conv2dHeNormal
      resnet.Linear = Linear
      if not getenv("SYNCBN"): resnet.BatchNorm = functools.partial(UnsyncedBatchNorm, num_devices=1)
      self.model = resnet.ResNet50()
      self.layers = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]

    layer = self.layers[layer_i][slice_i]
    xy = 112 >> layer_i
    if layer_i > 0: xy >>= (1 if slice_i > 0 else 0)
    name = f"layer{layer_i+1} slice{slice_i+1}"

    # get specific conv (0 or 1)
    if conv:
      if bn: f = lambda x: layer.bn2(layer.conv2(x))
      else: f = layer.conv2
      cin = layer.conv2.in_channels
      xy = xy // layer.conv1.stride
      return f"{name} conv2 x{(bs, cin, xy, xy)} k{layer.conv1.weight.shape}" + (" bn" if bn else ""), f, cin, xy

    cin = layer.conv1.in_channels
    return f"{name} x{(bs, cin, xy, xy)}", layer, cin, xy
  def _test_layer(self, name, layer, cin, xy):
    optim = SGD(get_parameters(layer), bs / 128 * 1.0)  # need sgd for some params but not consequential for benchmarking

    JITCNT = getenv("JITCNT", 1)
    @TinyJit
    def step(x):
      for _ in range(JITCNT):
        y = layer(x).relu()
        y.mean().backward()
        optim.step([y, x.grad])
      return y.detach()

    CNT = getenv("CNT", 5)
    best_tm = None
    flops = None
    for i in range(CNT):
      x = Tensor.randn(bs, cin, xy, xy, requires_grad=True).realize()
      GlobalCounters.reset()

      st = time.perf_counter()
      step(x)._data()
      et = time.perf_counter()

      if flops is None: flops = GlobalCounters.global_ops
      tm = (et-st) / JITCNT
      best_tm = tm if best_tm is None or tm < best_tm else best_tm
    print(f"\r{name:42s}: {best_tm * 1000: 4.2f} ms, {flops / 10**12 / tm: 3.2f} tflops")

  def test_layer1_1(self): self._test_layer(*self._get_layer(0, 0))
  def test_layer1_2(self): self._test_layer(*self._get_layer(0, 1))

  def test_layer2_1(self): self._test_layer(*self._get_layer(1, 0))

  def test_layer2_2(self): self._test_layer(*self._get_layer(1, 1))

  def test_layer3_1(self): self._test_layer(*self._get_layer(2, 0))

  def test_layer3_2(self): self._test_layer(*self._get_layer(2, 1))

  def test_layer4_1(self): self._test_layer(*self._get_layer(3, 0))

  def test_layer4_2(self): self._test_layer(*self._get_layer(3, 1))

if __name__ == '__main__':
  unittest.main()
