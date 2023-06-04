#!/usr/bin/env python
import os
if "OPT" not in os.environ:
  os.environ["OPT"] = "2"

import gc
import numpy as np

import unittest
from tinygrad.tensor import Tensor, Device
from tinygrad import nn
from tinygrad.helpers import getenv
from tinygrad.nn import optim
from tinygrad.ops import GlobalCounters, MovementOps, ReduceOps
from tinygrad.lazy import PUSH_PERMUTES

class CLCache():
  def __init__(self, allowed=None, strict=False, preclear=True): self.allowed, self.strict, self.preclear = allowed, strict, preclear
  def __enter__(self):
    if self.preclear:
      gc.collect()
      for x in [x for x in gc.get_objects() if isinstance(x, Tensor)]:
        x.realize()
      GlobalCounters.reset()
    GlobalCounters.cache = []
    print("cache: entering")
  def __exit__(self, type, value, traceback):
    print(f"cache: exiting with size {len(GlobalCounters.cache)}", f"allowed {self.allowed}" if self.allowed is not None else "")
    if self.allowed is not None:
      assert len(GlobalCounters.cache) <= self.allowed and (not self.strict or len(GlobalCounters.cache) == self.allowed), f"used too many kernels! {len(GlobalCounters.cache)} > {self.allowed}"
    GlobalCounters.cache = None

from models.convnext import ConvNeXt
from models.efficientnet import EfficientNet
from models.resnet import ResNet18
from models.vit import ViT
from tinygrad.nn.optim import get_parameters

@unittest.skipUnless(Device.DEFAULT == "GPU", "Not Implemented")
class TestInferenceMinKernels(unittest.TestCase):
  def setUp(self):
    Tensor.training = False

  @unittest.skipIf(not PUSH_PERMUTES, "this test requires PUSH_PERMUTES")
  def test_convnext(self):
    model = ConvNeXt()
    for p in get_parameters(model): p.assign(np.zeros(p.shape, dtype=p.dtype.np))
    img = Tensor.randn(1, 3, 224, 224)
    with CLCache(129):
      model(img).realize()

  def test_enet(self):
    model = EfficientNet(getenv("ENET_NUM", 0), has_se=False)
    for p in get_parameters(model): p.assign(np.zeros(p.shape, dtype=p.dtype.np))
    img = Tensor.randn(1, 3, 224, 224)
    with CLCache(51):
      model.forward(img).realize()

  def test_enet_se(self):
    model = EfficientNet(getenv("ENET_NUM", 0), has_se=True)
    for p in get_parameters(model): p.assign(np.zeros(p.shape, dtype=p.dtype.np))
    img = Tensor.randn(1, 3, 224, 224)
    # TODO: this seems very high
    with CLCache(115):
      model.forward(img).realize()

  def test_resnet(self):
    model = ResNet18()
    for p in get_parameters(model): p.assign(np.zeros(p.shape, dtype=p.dtype.np))
    img = Tensor.randn(1, 3, 224, 224)
    with CLCache(26):
      model.forward(img).realize()

  def test_vit(self):
    model = ViT(embed_dim=192, num_heads=3)
    for p in get_parameters(model): p.assign(np.zeros(p.shape, dtype=p.dtype.np))
    img = Tensor.randn(1, 3, 224, 224)
    with CLCache(223): # NOTE: this is way too high
      out = model.forward(img)
      assert len(GlobalCounters.cache) == 0, f"ViT prerealized?"
      out.realize()

  def test_llama(self):
    from examples.llama import Transformer
    args_tiny = {"dim": 512, "multiple_of": 256, "n_heads": 8, "n_layers": 4, "norm_eps": 1e-05, "vocab_size": 1000}
    model = Transformer(**args_tiny)
    for p in get_parameters(model): p.assign(np.zeros(p.shape, dtype=p.dtype.np))
    with CLCache(85):
      model(Tensor([[1,2,3,4]]), 0).realize()

@unittest.skipUnless(Device.DEFAULT == "GPU", "Not Implemented")
class TestOptBinOp(unittest.TestCase):
  def _test_no_binop_rerun(self, f1, f2=None, allowed=1):
    a = Tensor.randn(16, 16)
    b = Tensor.randn(16, 16)
    with CLCache():
      c = f1(a, b)
      if f2 is not None: d = f2(a, b)
      c.realize()
      if f2 is not None: d.realize()
      assert len(GlobalCounters.cache) == allowed, "binop was rerun!"
    if f2 is not None: np.testing.assert_allclose(c.numpy().ravel(), d.numpy().ravel(), rtol=1e-3, atol=1e-5)

  def test_no_binop_rerun(self): return self._test_no_binop_rerun(lambda a,b: a*b, lambda a,b: (a*b).reshape(16, 16, 1))
  def test_no_binop_rerun_alt(self): return self._test_no_binop_rerun(lambda a,b: (a*b).reshape(16, 16, 1), lambda a,b: a*b)
  def test_no_binop_rerun_reduce_broadcast(self): return self._test_no_binop_rerun(lambda a,b: a.sum()+b, lambda a,b: a.sum().reshape(1,1)+b, allowed=2)
  def test_no_binop_rerun_transposed(self): return self._test_no_binop_rerun(lambda a,b: (a.T*b.T).T, lambda a,b: a*b)
  def test_no_binop_rerun_mid_reshape(self): return self._test_no_binop_rerun(lambda a,b: (a*b).reshape(256)+a.reshape(256))

  # currently non working tests
  #def test_no_binop_rerun_preshape(self): return self._test_no_binop_rerun(lambda a,b: a.reshape(16, 16, 1)*b.reshape(16, 16, 1), lambda a,b: a*b)
  #def test_no_binop_rerun_reduce(self): return self._test_no_binop_rerun(lambda a,b: (a*b).sum(), lambda a,b: (a*b).reshape(16, 16, 1).sum())
  #def test_no_binop_rerun_reduce_alt(self): return self._test_no_binop_rerun(lambda a,b: a.sum(1)+b[0], lambda a,b: a.sum(1).reshape(1,16)+b[0])

@unittest.skip("elementwise with >1 reduce inputs currently don't fuse")
@unittest.skipUnless(Device.DEFAULT == "GPU", "Not Implemented")
class TestOptReduceLoop(unittest.TestCase):
  def test_loop_left(self):
    a = Tensor.randn(16, 16)
    b = Tensor.randn(16, 16)
    with CLCache():
      t = a.sum(0)
      b = t.reshape(16,1).expand(16,16).sum(0)
      c = (t+b)
      c.realize()
      assert len(GlobalCounters.cache) == 2, "loop left fusion broken"

  def test_loop_right(self):
    a = Tensor.randn(16, 16)
    b = Tensor.randn(16, 16)
    with CLCache():
      t = a.sum(0)
      b = t.reshape(16,1).expand(16,16).sum(0)
      c = (b+t)
      c.realize()
      assert len(GlobalCounters.cache) == 2, "loop right fusion broken"

@unittest.skipUnless(Device.DEFAULT == "GPU", "Not Implemented")
class TestOptWChild(unittest.TestCase):
  def test_unrealized_child(self):
    a = Tensor.randn(16, 16)
    b = Tensor.randn(16, 16)
    with CLCache():
      c = (a*b).sum()
      d = c+1
      e = c+2
      d.realize()
      assert len(GlobalCounters.cache) == 2, "don't fuse if you have children"

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
    # TODO: with Tensor.training
    Tensor.training = True
    img = Tensor.ones(1,32,4,4)
    bn = nn.BatchNorm2d(32, track_running_stats=False)
    with CLCache():
      img_bn = bn(img).realize()
      print(img_bn)
      assert len(GlobalCounters.cache) == 3, f"optimizer didn't fold batchnorm, got {len(GlobalCounters.cache)}"
    Tensor.training = False

  def test_fold_conv_sgd(self):
    # TODO: with Tensor.training
    Tensor.training = True
    img = Tensor.ones(2,3,4,4)
    c1 = nn.Conv2d(3,32,3)
    opt = optim.SGD(optim.get_parameters(c1))
    with CLCache():
      opt.zero_grad()
      c1(img).relu().sum().backward()
      opt.step()
      # TODO: this should be 4, but the sum output child stays around
      # with pushing_permutes it can be 3
      # TODO: broken with optim fixes
      assert len(GlobalCounters.cache) in [4,5,6], f"optimizer didn't fold conv-backward SGD, got {len(GlobalCounters.cache)}"
    Tensor.training = False

  def test_fold_2convs_sgd(self):
    # TODO: with Tensor.training
    Tensor.training = True
    img = Tensor.ones(2,3,64,64)
    c1 = nn.Conv2d(3,16,3,bias=False)
    c2 = nn.Conv2d(16,32,3,bias=False)
    opt = optim.SGD(optim.get_parameters([c1, c2]))
    with CLCache(allowed=9):
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      opt.step()
    Tensor.training = False

  def test_fold_4convs_sgd(self):
    # TODO: with Tensor.training
    Tensor.training = True
    img = Tensor.ones(2,3,64,64)
    c1 = nn.Conv2d(3,4,3,bias=False)
    c2 = nn.Conv2d(4,8,3,bias=False)
    c3 = nn.Conv2d(8,16,3,bias=False)
    c4 = nn.Conv2d(16,32,3,bias=False)
    opt = optim.SGD(optim.get_parameters([c1, c2, c3, c4]))
    with CLCache(allowed=19):
      opt.zero_grad()
      c4(c3(c2(c1(img).relu()).relu()).relu()).relu().sum().backward()
      opt.step()
    Tensor.training = False

  def test_fold_conv_batchnorm_sgd(self):
    # TODO: with Tensor.training
    Tensor.training = True
    img = Tensor.ones(1,3,4,4)
    c1 = nn.Conv2d(3,32,3)
    bn = nn.BatchNorm2d(32, track_running_stats=False)
    opt = optim.SGD(optim.get_parameters([c1, bn]))
    with CLCache(allowed=18): # this is too high
      img_bn = bn(c1(img)).elu().sum()
      opt.zero_grad()
      img_bn.backward()
      opt.step()
    Tensor.training = False

  def test_fold_conv_batchnorm_notrain(self):
    img = Tensor.ones(1,3,8,8)
    c1 = nn.Conv2d(3,32,3)
    bn = nn.BatchNorm2d(32, track_running_stats=False)
    # precache the bn
    img_conv = bn(c1(img)).relu().realize()
    with CLCache():
      img_conv = bn(c1(img)).relu().realize()
      assert len(GlobalCounters.cache) == 1, f"optimizer didn't fold conv-batchnorm at test time, got {len(GlobalCounters.cache)}"

  def test_fold_conv_batchnorm(self):
    Tensor.training = True
    img = Tensor.ones(1,3,8,8)
    c1 = nn.Conv2d(3,32,3)
    bn = nn.BatchNorm2d(32, track_running_stats=False)
    with CLCache():
      img_conv = bn(c1(img)).relu().realize()
      print(img_conv)
      assert len(GlobalCounters.cache) == 4, f"optimizer didn't fold conv-batchnorm, got {len(GlobalCounters.cache)}"
    Tensor.training = False

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

  def test_permute_was_pushed(self):
    a = Tensor.randn(16, 16, 16)
    with CLCache():
      c = a.sum(2)
      d = c.permute(1,0).contiguous()
      d.realize()
      cache_len = len(GlobalCounters.cache)
    np.testing.assert_allclose(a.numpy().sum(2).transpose(1,0), d.numpy(), rtol=1e-3, atol=1e-5)
    if PUSH_PERMUTES: assert cache_len == 1, "permute wasn't pushed!"

  def test_permute_was_pushed_though_contract_reshape(self):
    a = Tensor.randn(4, 4, 4, 4, 4)
    with CLCache():
      c = a.sum(-1)
      d = c.reshape(16,16).permute(1,0).contiguous()
      d.realize()
      cache_len = len(GlobalCounters.cache)
    np.testing.assert_allclose(a.numpy().sum(-1).reshape(16,16).transpose(1,0), d.numpy(), rtol=1e-3, atol=1e-5)
    if PUSH_PERMUTES: assert cache_len == 1, "permute wasn't pushed!"

  def test_permute_was_pushed_though_contractw1s_reshape(self):
    a = Tensor.randn(4, 4, 4, 4, 4)
    with CLCache():
      c = a.sum(-1)
      d = c.reshape(16,1,16).permute(2,1,0).contiguous()
      d.realize()
      cache_len = len(GlobalCounters.cache)
    np.testing.assert_allclose(a.numpy().sum(-1).reshape(16,1,16).transpose(2,1,0), d.numpy(), rtol=1e-3, atol=1e-5)
    if PUSH_PERMUTES: assert cache_len == 1, "permute wasn't pushed!"

  # TODO: push permute through expansion reshape
  @unittest.skip("expansion can't push expand permute yet")
  @unittest.skipIf(not PUSH_PERMUTES, "this test requires PUSH_PERMUTES")
  def test_permute_was_pushed_through_expand_reshape(self):
    a = Tensor.randn(16, 16, 16)
    with CLCache():
      c = a.sum(2)
      d = c.reshape(4,4,4,4).permute(2,3,0,1).contiguous()
      d.realize()
      cache_len = len(GlobalCounters.cache)
    np.testing.assert_allclose(a.numpy().sum(2).transpose(1,0).reshape(4,4,4,4), d.numpy(), rtol=1e-3, atol=1e-5)
    if PUSH_PERMUTES: assert cache_len == 1, "permute wasn't pushed!"

  @unittest.skipIf(PUSH_PERMUTES, "this test is broken with PUSH_PERMUTES")
  def test_no_reduceop_rerun(self):
    a = Tensor.randn(16, 16, 16)
    with CLCache():
      c = a.sum(2)
      d = a.sum(2).permute(1,0)
      c.realize()
      d.realize()
      cache_len = len(GlobalCounters.cache)
    np.testing.assert_allclose(c.numpy().transpose(1,0), d.numpy(), rtol=1e-3, atol=1e-5)
    assert cache_len == 1, "reduceop was rerun!"

  @unittest.skipIf(PUSH_PERMUTES, "this test is brokem with PUSH_PERMUTES")
  def test_no_reduceop_rerun_alt(self):
    a = Tensor.randn(16, 16, 16)
    with CLCache():
      c = a.sum(2).permute(1,0)
      d = a.sum(2)
      c.realize()
      d.realize()
      cache_len = len(GlobalCounters.cache)
    np.testing.assert_allclose(c.numpy(), d.numpy().transpose(1,0), rtol=1e-3, atol=1e-5)
    assert cache_len == 1, "reduceop was rerun!"

  def test_fold_with_contiguous(self):
    a = Tensor.randn(16, 16, 16)
    b = Tensor.randn(16, 16)
    with CLCache():
      c = (a.sum(2).contiguous() + b).contiguous()
      c.realize()
      cache_len = len(GlobalCounters.cache)
    assert cache_len == 1, "contiguous wasn't folded"

if __name__ == '__main__':
  unittest.main()
