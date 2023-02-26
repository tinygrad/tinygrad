#!/usr/bin/env python
import os
if "OPT" not in os.environ:
  os.environ["OPT"] = "2"

import gc
import numpy as np

import unittest
from tinygrad.tensor import Tensor, Device
from tinygrad import nn
from tinygrad.nn import optim
from tinygrad.ops import GlobalCounters

class CLCache():
  def __enter__(self):
    gc.collect()
    for x in [x for x in gc.get_objects() if isinstance(x, Tensor)]:
      x.realize()
    GlobalCounters.cache = []
    print("cache: entering")
  def __exit__(self, type, value, traceback):
    print(f"cache: exiting with size {len(GlobalCounters.cache)}")
    GlobalCounters.cache = None

@unittest.skipUnless(Device.DEFAULT == "GPU", "Not Implemented")
class TestOpt(unittest.TestCase):
  def test_muladd(self):
    a,b,c = [Tensor.ones(2,2) for _ in range(3)]
    with CLCache():
      d = a * b + c
      d.realize()
      assert len(GlobalCounters.cache) == 1, "optimizer didn't fold muladd"
    np.testing.assert_allclose(d.numpy(), np.ones((2,2))*2, rtol=1e-5)

  def test_fold_reduce_elementwise(self):
    img = Tensor.ones(32)
    addme = Tensor.ones(1)
    with CLCache():
      ret = img.sum() + addme
      ret.realize()
      assert len(GlobalCounters.cache) == 1, "optimizer didn't fold reduce/elementwise"
    assert ret.numpy()[0] == 33

  def test_fold_batchnorm(self):
    with Tensor.train():
      img = Tensor.ones(1,32,4,4)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      with CLCache():
        img_bn = bn(img).realize()
        print(img_bn)
        assert len(GlobalCounters.cache) == 3, "optimizer didn't fold batchnorm"

  def test_fold_conv_sgd(self):
    with Tensor.train():
      img = Tensor.ones(1,3,4,4)
      c1 = nn.Conv2d(3,32,3)
      opt = optim.SGD(optim.get_parameters(c1))
      with CLCache():
        opt.zero_grad()
        c1(img).relu().sum().backward()
        opt.step()
        # TODO: this should be 4, but the sum output child stays around
        # with pushing_permutes it can be 3
        assert len(GlobalCounters.cache) in [4,5], "optimizer didn't fold conv-backward SGD"

  def test_fold_conv_batchnorm_sgd(self):
    with Tensor.train():
      img = Tensor.ones(1,3,4,4)
      c1 = nn.Conv2d(3,32,3)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      opt = optim.SGD(optim.get_parameters([c1, bn]))
      with CLCache():
        img_bn = bn(c1(img)).elu().sum()
        opt.zero_grad()
        img_bn.backward()
        opt.step()
        assert len(GlobalCounters.cache) in [9,10], "optimizer didn't fold conv-backward batchnorm"

  def test_fold_conv_batchnorm_notrain(self):
    img = Tensor.ones(1,3,8,8)
    c1 = nn.Conv2d(3,32,3)
    bn = nn.BatchNorm2d(32, track_running_stats=False)
    # precache the bn
    img_conv = bn(c1(img)).relu().realize()
    with CLCache():
      img_conv = bn(c1(img)).relu().realize()
      assert len(GlobalCounters.cache) == 1, "optimizer didn't fold conv-batchnorm at test time"

  def test_fold_conv_batchnorm(self):
    with Tensor.train():
      img = Tensor.ones(1,3,8,8)
      c1 = nn.Conv2d(3,32,3)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      with CLCache():
        img_conv = bn(c1(img)).relu().realize()
        print(img_conv)
        assert len(GlobalCounters.cache) == 4, "optimizer didn't fold conv-batchnorm"

  def test_fold_conv_elu(self):
    img = Tensor.ones(1,4,8,8)
    c1 = nn.Conv2d(4, 4, kernel_size=3)
    c2 = nn.Conv2d(4, 4, kernel_size=3)
    with CLCache():
      img_conv = img.sequential([c1, Tensor.elu, c2, Tensor.elu]).realize()
      print(img_conv)
      assert len(GlobalCounters.cache) == 2, "optimizer didn't fold conv/elu"

  def test_fold_conv_relu(self):
    img = Tensor.ones(1,4,8,8)
    c1 = nn.Conv2d(4, 4, kernel_size=3)
    c2 = nn.Conv2d(4, 4, kernel_size=3)
    with CLCache():
      img_conv = img.sequential([c1, Tensor.relu, c2, Tensor.relu]).realize()
      print(img_conv)
      assert len(GlobalCounters.cache) == 2, "optimizer didn't fold conv/relu"

  def test_fold_conv_relu_nobias(self):
    img = Tensor.ones(1,4,8,8)
    c1 = nn.Conv2d(4, 4, kernel_size=3, bias=False)
    c2 = nn.Conv2d(4, 4, kernel_size=3, bias=False)
    with CLCache():
      img_conv = img.sequential([c1, Tensor.relu, c2, Tensor.relu]).realize()
      print(img_conv)
      assert len(GlobalCounters.cache) == 2, "optimizer didn't fold conv/relu"

if __name__ == '__main__':
  unittest.main()
