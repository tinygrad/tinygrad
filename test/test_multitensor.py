import unittest, functools
from tinygrad import Tensor, Device, nn, GlobalCounters, TinyJit
from tinygrad.device import _BufferCopy
from tinygrad.ops import LoadOps
from tinygrad.helpers import CI
from tinygrad.nn.state import get_parameters
import numpy as np

d_zero = f"{Device.DEFAULT}:0"
d0, d1 = f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2"
d2, d3 = f"{Device.DEFAULT}:3", f"{Device.DEFAULT}:4"
devices_2 = (d0, d1)
devices_3 = (d0, d1, d2)
N = 128

# shard_x is "data parallel"
# shard_w is "model parallel"

@unittest.skipIf(CI and Device.DEFAULT in {"GPU", "CUDA", "METAL"}, "no GPU CI")
class TestMultiTensor(unittest.TestCase):
  def test_to(self):
    X = Tensor.ones(256).contiguous().realize()
    X.to_((d0, d1))
    for lb in X.lazydata.lbs:
      assert lb.shape == (256,)
    (X + X).realize()

  def test_shard(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_((d0, d1), 0)
    for lb in X.lazydata.lbs:
      assert lb.shape == (128,)
    (X + X).realize()

  def test_shard_same_device(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_((d0, X.device), 0)
    (X + X).realize()

  def test_shard_plus_one_sum(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_([d0, d1], 0)
    (X + 1).sum().realize()

  def test_shard_plus_one_sum_d_zero(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_([d_zero, d1], 0)
    (X + 1).sum().realize()

  def test_numpy(self):
    X = Tensor.ones(256)
    X.shard_((d0, d1), 0)
    np.testing.assert_allclose(X.numpy(), 1)

  def _test_simple_add_axis(self, shard_x, shard_w):
    X = Tensor.ones(256).contiguous().realize()
    W = Tensor.ones(256).contiguous().realize()
    X.shard_((d0, d1), shard_x)
    W.shard_((d0, d1), shard_w)
    O = X + W
    np.testing.assert_allclose(O.numpy(), 2)

  def test_simple_add(self): return self._test_simple_add_axis(None, None)
  def test_simple_add_X(self): return self._test_simple_add_axis(0, None)
  def test_simple_add_W(self): return self._test_simple_add_axis(None, 0)
  def test_simple_add_XW(self): return self._test_simple_add_axis(0, 0)

  def test_four_add(self):
    X = Tensor.ones(256, 256).contiguous().realize()
    W = Tensor.ones(256, 256).contiguous().realize()
    X.shard_((d0, d1, d2, d3), 1)
    W.shard_((d0, d1, d2, d3), None)
    O = X + W
    np.testing.assert_allclose(O.numpy(), 2)

  def _test_sum_axis(self, shard_x):
    X = Tensor.ones(256, 256).contiguous().realize()
    X.shard_((d0, d1), shard_x)
    O = X.sum(axis=0)
    np.testing.assert_allclose(O.numpy(), 256)
    O = X.sum(axis=1)
    np.testing.assert_allclose(O.numpy(), 256)
    O = X.sum()
    np.testing.assert_allclose(O.numpy(), 256*256)

  def test_sum(self): return self._test_sum_axis(None)
  def test_sum_0(self): return self._test_sum_axis(0)
  def test_sum_1(self): return self._test_sum_axis(1)

  def _test_max_axis(self, shard_x, sign=1):
    X = Tensor.arange(16).reshape(4, 4) * sign
    n = X.numpy()
    X.shard_((d0, d1), shard_x)
    O = X.max(axis=0)
    np.testing.assert_allclose(O.numpy(), n.max(0))
    O = X.max(axis=1)
    np.testing.assert_allclose(O.numpy(), n.max(1))
    O = X.max()
    np.testing.assert_allclose(O.numpy(), n.max())

  def test_max(self): return self._test_max_axis(None)
  def test_max_0(self): return self._test_max_axis(0)
  def test_max_1(self): return self._test_max_axis(1)
  def test_max_neg(self): return self._test_max_axis(None, sign=-1)
  def test_max_0_neg(self): return self._test_max_axis(0, sign=-1)
  def test_max_1_neg(self): return self._test_max_axis(1, sign=-1)

  def _test_matmul_shard_axis(self, shard_x, shard_w, device):
    X = Tensor.kaiming_uniform(N, N).realize()
    W = Tensor.kaiming_uniform(N, N).realize()
    Xs = X.shard(device, shard_x)
    Ws = W.shard(device, shard_w)
    O = (Xs@Ws)
    np.testing.assert_allclose(X.numpy() @ W.numpy(), O.to(Device.DEFAULT).numpy(), atol=1e-5)

  def _test_double_matmul_shard_axis(self, shard_x, shard_w, device):
    X = Tensor.kaiming_uniform(N, N).realize()
    W1 = Tensor.kaiming_uniform(N, N).realize()
    W2 = Tensor.kaiming_uniform(N, N).realize()
    Xs = X.shard(device, shard_x)
    W1s = W1.shard(device, shard_w)
    W2s = W2.shard(device, shard_w)
    O = (Xs@W1s)@W2s
    np.testing.assert_allclose((X.numpy() @ W1.numpy()) @ W2.numpy(), O.to(Device.DEFAULT).numpy(), atol=1e-5)

  def test_matmul_shard_none(self): return self._test_matmul_shard_axis(None, None, devices_2)
  def test_matmul_shard_X_0(self): return self._test_matmul_shard_axis(0, None, devices_2)
  def test_matmul_shard_X_1(self): return self._test_matmul_shard_axis(1, None, devices_2)
  def test_matmul_shard_W_0(self): return self._test_matmul_shard_axis(None, 0, devices_2)
  def test_matmul_shard_W_1(self): return self._test_matmul_shard_axis(None, 1, devices_2)

  def test_matmul_shard_0_0(self): return self._test_matmul_shard_axis(0, 0, devices_2)
  def test_matmul_shard_0_1(self): return self._test_matmul_shard_axis(0, 1, devices_2)
  def test_matmul_shard_1_0(self): return self._test_matmul_shard_axis(1, 0, devices_2)
  def test_matmul_shard_1_1(self): return self._test_matmul_shard_axis(1, 1, devices_2)

  def test_double_matmul_shard_X_0(self): return self._test_double_matmul_shard_axis(0, None, devices_2)
  def test_double_matmul_shard_X_1(self): return self._test_double_matmul_shard_axis(1, None, devices_2)
  def test_double_matmul_shard_W_0(self): return self._test_double_matmul_shard_axis(None, 0, devices_2)
  def test_double_matmul_shard_W_1(self): return self._test_double_matmul_shard_axis(None, 1, devices_2)

  def test_conv_data_shard(self):
    conv = nn.Conv2d(3, 16, 3, bias=False)
    for p in get_parameters(conv): p.shard_((d0, d1))
    fake_image = Tensor.rand((2, 3, 32, 32)).shard((d0, d1), axis=0)
    out = conv(fake_image)
    out.numpy()

  def test_conv_bias_data_shard(self):
    conv = nn.Conv2d(3, 16, 3)
    for p in get_parameters(conv): p.shard_((d0, d1))
    fake_image = Tensor.rand((2, 3, 32, 32)).shard((d0, d1), axis=0)
    out = conv(fake_image)
    out.numpy()

  def test_backprop_conv(self):
    conv = nn.Conv2d(3, 16, 3)
    for p in get_parameters(conv): p.shard_((d0, d1))
    optim = nn.optim.Adam(get_parameters(conv))
    fake_image = Tensor.rand((2, 3, 32, 32)).shard((d0, d1), axis=0)
    out = conv(fake_image)
    optim.zero_grad()
    out.mean().backward()
    #for p in get_parameters(conv): p.grad.realize()
    optim.step()

  def test_lr_scheduler_OneCycleLR(self):
    from extra.lr_scheduler import OneCycleLR
    conv = nn.Conv2d(3, 16, 3)
    for p in get_parameters(conv): p.shard_((d0, d1))
    optim = nn.optim.SGD(get_parameters(conv))
    lr_sched = OneCycleLR(optim, max_lr=0.1, pct_start=0.1, div_factor=100, final_div_factor=0.1, total_steps=10)
    lr_sched.step()

  def test_embedding(self):
    B, T, embed_size, vocab_size = 4, 10, 20, 28

    layer = nn.Embedding(vocab_size, embed_size)
    x = Tensor(np.random.randint(0, vocab_size, (B, T)))
    z = layer(x)

    layer_sharded = nn.Embedding(vocab_size, embed_size)
    layer_sharded.weight.assign(layer.weight.shard((d0, d1), axis=1)).realize()
    x_sharded = x.shard((d0, d1), axis=None)
    z_shard = layer_sharded(x_sharded)

    np.testing.assert_allclose(z.numpy(), z_shard.numpy(), atol=1e-6, rtol=1e-6)

  def test_rmsnorm(self):
    from extra.models.llama import RMSNorm
    B, T, embed_size = 4, 10, 20

    layer_norm = RMSNorm(embed_size)
    x = Tensor.rand((B, T, embed_size)).contiguous().realize()
    y = layer_norm(x)

    # for norm layers, the correct way to shard weights is duplication
    layer_norm_sharded = RMSNorm(embed_size)
    layer_norm_sharded.weight.shard_((d0, d1), axis=None).realize()

    # if x is being sharded, then all-reduce is involved
    x_sharded = x.shard((d0, d1), axis=2).realize()
    y_shard = layer_norm_sharded(x_sharded).realize()
    np.testing.assert_allclose(y.numpy(), y_shard.numpy(), atol=1e-6, rtol=1e-6)

    # if x is being duplicated, then the operations remain inside each GPU
    # which is the common case
    x_sharded = x.shard((d0, d1), axis=None).realize()
    y_shard = layer_norm_sharded(x_sharded).realize()
    np.testing.assert_allclose(y.numpy(), y_shard.numpy(), atol=1e-6, rtol=1e-6)

  def test_data_parallel_resnet(self):
    import sys, pathlib
    sys.path.append((pathlib.Path(__file__).parent.parent / "extra" / "models").as_posix())
    from resnet import ResNet18

    fake_image = Tensor.rand((2, 3, 224, 224))
    fake_image_sharded = fake_image.shard((d0, d1), axis=0)
    m = ResNet18()
    m.load_from_pretrained()
    real_output = m(fake_image).numpy()
    for p in get_parameters(m): p.shard_((d0, d1)).realize()
    GlobalCounters.reset()
    shard_output = m(fake_image_sharded).realize()
    assert shard_output.lazydata.lbs[0].shape == (1, 1000)
    assert shard_output.lazydata.lbs[1].shape == (1, 1000)
    shard_output_np = shard_output.numpy()
    np.testing.assert_allclose(real_output, shard_output_np, atol=1e-6, rtol=1e-6)

  def test_multi_tensor_jit_param(self):
    @TinyJit
    def jf(a, b) -> Tensor:
      return (a + b).realize()

    for _ in range(5):
      a = Tensor.ones(256).contiguous().realize()
      b = Tensor.ones(256).contiguous().realize()
      a.shard_((d0, d1))
      b.shard_((d0, d1))
      c = jf(a, b)
      np.testing.assert_allclose(c.numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
    assert len(jf.jit_cache) > 0

  def test_multi_tensor_jit_body(self):
    @TinyJit
    def jf() -> Tensor:
      a = Tensor.ones(256).contiguous().realize()
      b = Tensor.ones(256).contiguous().realize()
      a.shard_((d0, d1))
      b.shard_((d0, d1))
      return (a + b).realize()

    for _ in range(5):
      r = jf()
      np.testing.assert_allclose(r.numpy(), np.ones(256)+np.ones(256), atol=1e-4, rtol=1e-5)
    assert len(jf.jit_cache) > 0

  @unittest.skipIf(CI and Device.DEFAULT=="METAL", "no ICB in CI, creation of graph fails")
  def test_multi_device_jit_graph(self):
    if Device[d0].graph is None or Device[d1].graph is None: raise unittest.SkipTest("only test graphs")

    @TinyJit
    def jf(a: Tensor, b: Tensor, c: Tensor, d:Tensor):
      # Create 80 entries on device 0: 2 batches.
      for _ in range(40):
        a = ((a + b).realize() + (a * b).realize()).realize()
      # Create 80 entries on device 1: 2 batches.
      for _ in range(40):
        c = ((c + d).realize() + (c * d).realize()).realize()
      # Create a copy from device 0 to 1: 1 entry.
      a = a.to(d1).realize()
      # Creates one last entry on device 1: 1 batch.
      return (a + c).realize()

    a = Tensor.randn(10, 10, device=d0).realize()
    b = Tensor.randn(10, 10, device=d0).realize()
    c = Tensor.randn(10, 10, device=d1).realize()
    d = Tensor.randn(10, 10, device=d1).realize()

    ref = jf(a, b, c, d).numpy()
    for _ in range(5):
      o = jf(a, b, c, d).numpy()
      np.testing.assert_allclose(ref, o, atol=1e-4, rtol=1e-5)

    graph_d0 = Device[d0].graph.func if isinstance(Device[d0].graph, functools.partial) else Device[d0].graph
    graph_d1 = Device[d1].graph.func if isinstance(Device[d1].graph, functools.partial) else Device[d1].graph
    # Checking that 2 graphs per device, 1 copy and 1 last graph on device 1 are created.
    assert isinstance(jf.jit_cache[0].prg, graph_d0)
    assert isinstance(jf.jit_cache[1].prg, graph_d0)
    assert isinstance(jf.jit_cache[2].prg, graph_d1)
    assert isinstance(jf.jit_cache[3].prg, graph_d1)
    assert isinstance(jf.jit_cache[4].prg, _BufferCopy)
    assert isinstance(jf.jit_cache[5].prg, graph_d1)

  def test_uneven_shard(self):
    for N in range(1, 6):
      X = Tensor.rand(4, 1, 257).contiguous().realize()
      n = X.numpy()
      devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(N))
      X.shard_(devices, 2)
      np.testing.assert_equal(X.numpy(), n)
      np.testing.assert_equal(X.reshape(2, 2, 257).numpy(), n.reshape((2, 2, 257)))
      np.testing.assert_equal(X.shrink(((0,2), (0, 1), (0,257))).numpy(), n[0:2, 0:1, 0:257])
      np.testing.assert_equal(X.expand((4, 4, 257)).numpy(), np.tile(n, (1, 4, 1)))
      np.testing.assert_equal(X.permute((0, 2, 1)).numpy(), np.transpose(n, (0, 2, 1)))

  def test_bn_ast_on_devices(self):
    devices = (d0, d1, d2, d3)
    t = Tensor.empty((16, 64, 112, 112)).shard(devices, axis=0)
    bn = nn.BatchNorm2d(64)
    for p in get_parameters(bn): p.shard_(devices).realize()

    out = bn(t)
    scheds = [sched for sched in out.lazydata.schedule() if sched.out.device in devices and sched.ast.op is not LoadOps.COPY]
    assert set(sched.out.device for sched in scheds) == set(devices), "should have ast on each shard device"
    asts = [sched.ast for sched in scheds]
    assert len(asts) == 4, len(asts)
    # test case to show that ast can be different on devices
    # TODO: make ast identical on devices
    assert len(set(asts)) == 4, len(asts)
    # for i, ast in enumerate(asts):
    #   print(f"{i} {ast}")

if __name__ == '__main__':
  unittest.main()