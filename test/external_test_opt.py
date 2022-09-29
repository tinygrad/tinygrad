#!/usr/bin/env python
import os
os.environ["LAZY"] = "1"
os.environ["OPT"] = "2"

import gc
import numpy as np

import unittest
from tinygrad.tensor import Tensor, Device
from tinygrad import nn
from tinygrad.nn import optim

try:
  from tinygrad.llops.ops_gpu import CL
except ImportError:
  pass

class CLCache():
  def __enter__(self):
    gc.collect()
    for x in [x for x in gc.get_objects() if isinstance(x, Tensor)]:
      x.realize()
    CL.CACHE = []
    print("cache: entering")
  def __exit__(self, type, value, traceback):
    print(f"cache: exiting with size {len(CL.CACHE)}")
    for prg, args in CL.CACHE:
      e = prg.clprg(CL().cl_queue, *args)
    CL.CACHE = None

@unittest.skipUnless(Device.DEFAULT == "GPU", "Not Implemented")
class TestOpt(unittest.TestCase):
  def test_muladd(self):
    a,b,c = [Tensor.ones(2,2) for _ in range(3)]
    with CLCache():
      d = a * b + c
      d.realize()
      assert len(CL.CACHE) == 1, "optimizer didn't fold muladd"
    np.testing.assert_allclose(d.numpy(), np.ones((2,2))*2, rtol=1e-5)

  def test_fold_reduce_elementwise(self):
    img = Tensor.ones(32)
    addme = Tensor.ones(1)
    with CLCache():
      ret = img.sum() + addme
      ret.realize()
      assert len(CL.CACHE) == 1, "optimizer didn't fold reduce/elementwise"
    assert ret.numpy()[0] == 33

  def test_fold_batchnorm(self):
    # TODO: with Tensor.training
    Tensor.training = True
    img = Tensor.ones(1,32,4,4)
    bn = nn.BatchNorm2D(32, track_running_stats=False)
    with CLCache():
      img_bn = bn(img).realize()
      print(img_bn)
      assert len(CL.CACHE) == 3, "optimizer didn't fold batchnorm"
    Tensor.training = False

  def test_fold_conv_batchnorm_sgd(self):
    # TODO: with Tensor.training
    Tensor.training = True
    img = Tensor.ones(1,3,4,4)
    c1 = nn.Conv2d(3,32,3)
    bn = nn.BatchNorm2D(32, track_running_stats=False)
    opt = optim.SGD(optim.get_parameters([c1, bn]))
    with CLCache():
      img_bn = bn(c1(img)).elu().sum()
      opt.zero_grad()
      img_bn.backward()
      opt.step()
      assert len(CL.CACHE) == 10, "optimizer didn't fold conv-backward batchnorm"
    Tensor.training = False

  def test_fold_conv_batchnorm_notrain(self):
    img = Tensor.ones(1,3,8,8)
    c1 = nn.Conv2d(3,32,3)
    bn = nn.BatchNorm2D(32, track_running_stats=False)
    # precache the bn
    img_conv = bn(c1(img)).relu().realize()
    with CLCache():
      img_conv = bn(c1(img)).relu().realize()
      assert len(CL.CACHE) == 1, "optimizer didn't fold conv-batchnorm at test time"

  def test_fold_conv_batchnorm(self):
    Tensor.training = True
    img = Tensor.ones(1,3,8,8)
    c1 = nn.Conv2d(3,32,3)
    bn = nn.BatchNorm2D(32, track_running_stats=False)
    with CLCache():
      img_conv = bn(c1(img)).relu().realize()
      print(img_conv)
      assert len(CL.CACHE) == 4, "optimizer didn't fold conv-batchnorm"
    Tensor.training = False

  def test_fold_conv_elu(self):
    img = Tensor.ones(1,4,8,8)
    c1 = nn.Conv2d(4, 4, kernel_size=3)
    c2 = nn.Conv2d(4, 4, kernel_size=3)
    with CLCache():
      img_conv = img.sequential([c1, Tensor.elu, c2, Tensor.elu]).realize()
      print(img_conv)
      assert len(CL.CACHE) == 2, "optimizer didn't fold conv/elu"

if __name__ == '__main__':
  unittest.main()