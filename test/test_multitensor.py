import unittest, functools, random
from typing import List
from tinygrad import Tensor, Device, nn, GlobalCounters, TinyJit, dtypes
from tinygrad.ops import MetaOps, ReduceOps, BinaryOps, UOps
from tinygrad.helpers import CI, getenv, prod, Context
from tinygrad.nn.state import get_parameters, get_state_dict
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import lower_schedule, BufferCopy, CompiledRunner
from tinygrad.multi import all_reduce, MultiLazyBuffer
import numpy as np
from hypothesis import given, strategies as strat, settings
from test.helpers import is_dtype_supported

settings.register_profile("my_profile", max_examples=200, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")

d0 = f"{Device.DEFAULT}:0"
d1 = f"{Device.DEFAULT}:1"
d2 = f"{Device.DEFAULT}:2"
d3 = f"{Device.DEFAULT}:3"
d4 = f"{Device.DEFAULT}:4"
d5 = f"{Device.DEFAULT}:5"
devices_2 = (d1, d2)
devices_3 = (d1, d2, d3)
devices_4 = (d1, d2, d3, d4)
N = 128

# shard_x is "data parallel"
# shard_w is "model parallel"

def _test_allreduce(t:Tensor):
  aa = (t[0:64] + t[64:128] + t[128:192] + t[192:256]).repeat([4,1]).realize()
  ts = t.shard(devices_4, 0).realize()
  b = Tensor(MultiLazyBuffer(all_reduce(ReduceOps.SUM, ts.lazydata.lbs), 0))
  b.realize()
  return aa, b

@unittest.skipIf(CI and Device.DEFAULT in ("GPU", "CUDA", "METAL"), "no GPU CI")
class TestMultiTensor(unittest.TestCase):
  def test_to(self):
    X = Tensor.ones(256).contiguous().realize()
    X.to_(devices_2)
    for lb in X.lazydata.lbs:
      assert lb.shape == (256,)
    (X + X).realize()

  def test_shard(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_(devices_2, 0)
    for lb in X.lazydata.lbs:
      assert lb.shape == (128,)
    (X + X).realize()

  def test_sharded_arange(self):
    sharded_arange = Tensor.arange(1000).shard(devices_2, 0)
    sharded_arange.realize()
    np.testing.assert_equal(sharded_arange.numpy(), np.arange(1000))

  def test_shard_no_recompile(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_(devices_2, 0)
    out = (X + X)
    sched = create_schedule(out.lazydata.lbs)
    names = []
    for si, ei in zip(sched[:], lower_schedule(sched)):
      if isinstance(ei.prg, CompiledRunner): names.append(ei.prg.p.name)
      ei.run()
    assert names[-2] == names[-1], "function was relinearized"

  def test_sharded_memory(self):
    # Buffer may be stuck in track_cross_buffer
    for x in (d0, d1, d2, d3, d4): Device[x].synchronize()
    mem_base = GlobalCounters.mem_used

    X = Tensor.ones(256).contiguous().realize()
    assert GlobalCounters.mem_used-mem_base== X.dtype.itemsize * 256, GlobalCounters.mem_used-mem_base
    X.shard_(devices_4).realize()
    for x in (d0, d1, d2, d3, d4): Device[x].synchronize()
    assert GlobalCounters.mem_used-mem_base == X.dtype.itemsize * 256 * 4, GlobalCounters.mem_used-mem_base

    X = Tensor.ones(256).contiguous().realize()
    assert GlobalCounters.mem_used-mem_base == X.dtype.itemsize * 256, GlobalCounters.mem_used-mem_base
    X.shard_(devices_4, axis=0).realize()
    for x in (d0, d1, d2, d3, d4): Device[x].synchronize()
    assert GlobalCounters.mem_used-mem_base == X.dtype.itemsize * 256, GlobalCounters.mem_used-mem_base

    X = Tensor.ones(256).realize()
    assert GlobalCounters.mem_used-mem_base == 0
    X.shard_(devices_4).realize()
    assert GlobalCounters.mem_used-mem_base == 0

    X = Tensor.ones(256).realize()
    assert GlobalCounters.mem_used-mem_base == 0
    X.shard_(devices_4, axis=0).realize()
    assert GlobalCounters.mem_used-mem_base == 0

  def test_shard_same_device(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_((d1, X.device), 0)
    (X + X).realize()

  def test_shard_plus_one_sum(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_((d1, d2), 0)
    (X + 1).sum().realize()

  def test_shard_plus_one_sum_d0(self):
    X = Tensor.ones(256).contiguous().realize()
    X.shard_((d0, d2), 0)
    (X + 1).sum().realize()

  def test_numpy(self):
    X = Tensor.ones(256)
    X.shard_((d1, d2), 0)
    np.testing.assert_allclose(X.numpy(), 1)

  def _test_simple_add_axis(self, shard_x, shard_w):
    X = Tensor.ones(256).contiguous().realize()
    W = Tensor.ones(256).contiguous().realize()
    X.shard_((d1, d2), shard_x)
    W.shard_((d1, d2), shard_w)
    O = X + W
    np.testing.assert_allclose(O.numpy(), 2)

  def test_simple_add(self): return self._test_simple_add_axis(None, None)
  def test_simple_add_X(self): return self._test_simple_add_axis(0, None)
  def test_simple_add_W(self): return self._test_simple_add_axis(None, 0)
  def test_simple_add_XW(self): return self._test_simple_add_axis(0, 0)

  def test_four_add(self):
    X = Tensor.ones(256, 256).contiguous().realize()
    W = Tensor.ones(256, 256).contiguous().realize()
    X.shard_(devices_4, 1)
    W.shard_(devices_4, None)
    O = X + W
    np.testing.assert_allclose(O.numpy(), 2)

  def test_elementwise_dtype(self):
    Tensor.manual_seed(0)
    X = Tensor.randn(8, 8).realize()
    W = Tensor.randn(8, 8).realize()
    X.shard_(devices_4, 0)
    W.shard_(devices_4, 0)
    O = X.shrink(((0, 2), None)) * W.shrink(((0, 2), None)) < 2
    np.testing.assert_allclose(O.numpy(), X.numpy()[0:2]*W.numpy()[0:2] < 2)

  @given(strat.sampled_from((4, 5)), strat.sampled_from((devices_2, devices_3)), strat.sampled_from((ReduceOps.SUM, ReduceOps.PROD, ReduceOps.MAX)),
         strat.sampled_from((None, 0, 1)), strat.sampled_from((None, 0, 1)), strat.sampled_from((1, 0, -1)))
  def test_simple_reduce(self, N, devices, rop, shard_axis, reduce_axis, sign):
    X = Tensor.rand(N*N).reshape(N, N).mul(sign)
    n = X.numpy()
    X.shard_(devices, shard_axis)
    f = {ReduceOps.SUM: lambda x: x.sum(reduce_axis), ReduceOps.PROD: lambda x: x.prod(reduce_axis), ReduceOps.MAX: lambda x: x.max(reduce_axis)}[rop]
    fX = f(X)
    fn = f(n)
    np.testing.assert_allclose(fX.numpy(), fn, rtol=1e-6, atol=1e-6)

  def test_allreduce_naive(self):
    with Context(RING=0):
      a,b = _test_allreduce(Tensor.rand(256, 256))
      np.testing.assert_almost_equal(a.numpy(), b.numpy(), decimal=5)

  def test_allreduce_ring(self):
    with Context(RING=2):
      a,b = _test_allreduce(Tensor.rand(256, 256))
      np.testing.assert_almost_equal(a.numpy(), b.numpy(), decimal=5)

  def test_copy_jit(self):
    @TinyJit
    def copy_tensor(x:Tensor): return (x.to(f"{x.device.split(':')[0]}:1") + 1)
    for _ in range(5):
      t = Tensor.rand(256).realize()
      x = copy_tensor(t)
      np.testing.assert_equal((t+1).numpy(), x.numpy())

  def test_allreduce_naive_jit(self):
    with Context(RING=0):
      jit_allreduce = TinyJit(_test_allreduce)
      for _ in range(5):
        a,b = jit_allreduce(Tensor.rand(256, 256))
        np.testing.assert_almost_equal(a.numpy(), b.numpy(), decimal=5)

  def test_allreduce_ring_jit(self):
    with Context(RING=2):
      jit_allreduce = TinyJit(_test_allreduce)
      for _ in range(5):
        a,b = jit_allreduce(Tensor.rand(256, 256))
        np.testing.assert_almost_equal(a.numpy(), b.numpy(), decimal=5)

  @unittest.skip("slow")
  def test_fuzz_allreduce(self):
    random.seed(41)
    for it in range(100):
      for n in range(2, 4+1):
        shape = tuple([(n if i == 0 else 1) * random.randint(1, 10) for i in range(random.randint(1, 4))])
        t = Tensor.rand(shape).shard_(tuple([d0, d1, d2, d3][:n]), 0)
        with Context(RING=0):
          a = Tensor(MultiLazyBuffer(all_reduce(ReduceOps.SUM, t.lazydata.lbs), 0))
        with Context(RING=2):
          b = Tensor(MultiLazyBuffer(all_reduce(ReduceOps.SUM, t.lazydata.lbs), 0))
        diff = a - b
        mean_err = diff.reshape((prod(diff.shape),)).abs().mean().numpy()
        max_err = diff.reshape((prod(diff.shape),)).abs().max().numpy()
        assert mean_err < 1e-6, f"big mean error, iteration {it}_{n}"
        assert max_err < 1e-6, f"big max error, iteration {it}_{n}"

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
    for p in get_parameters(conv): p.shard_(devices_2)
    fake_image = Tensor.rand((2, 3, 32, 32)).shard(devices_2, axis=0)
    out = conv(fake_image)
    out.numpy()

  def test_conv_bias_data_shard(self):
    conv = nn.Conv2d(3, 16, 3)
    for p in get_parameters(conv): p.shard_(devices_2)
    fake_image = Tensor.rand((2, 3, 32, 32)).shard(devices_2, axis=0)
    out = conv(fake_image)
    out.numpy()

  def test_backprop_conv(self):
    with Tensor.train():
      conv = nn.Conv2d(3, 16, 3)
      for p in get_parameters(conv): p.shard_(devices_2)
      optim = nn.optim.Adam(get_parameters(conv))
      fake_image = Tensor.rand((2, 3, 32, 32)).shard(devices_2, axis=0)
      out = conv(fake_image)
      optim.zero_grad()
      out.mean().backward()
      #for p in get_parameters(conv): p.grad.realize()
      optim.step()
      out.numpy()

  def test_lr_scheduler_OneCycleLR(self):
    from extra.lr_scheduler import OneCycleLR
    conv = nn.Conv2d(3, 16, 3)
    for p in get_parameters(conv): p.shard_(devices_2)
    optim = nn.optim.SGD(get_parameters(conv))
    lr_sched = OneCycleLR(optim, max_lr=0.1, pct_start=0.1, div_factor=100, final_div_factor=0.1, total_steps=10)
    lr_sched.step()

  def test_embedding(self):
    B, T, embed_size, vocab_size = 4, 10, 20, 28

    layer = nn.Embedding(vocab_size, embed_size)
    x = Tensor(np.random.randint(0, vocab_size, (B, T)))
    z = layer(x)

    layer_sharded = nn.Embedding(vocab_size, embed_size)
    layer_sharded.weight.replace(layer.weight.shard(devices_2, axis=1)).realize()
    x_sharded = x.shard(devices_2, axis=None)
    z_shard = layer_sharded(x_sharded)

    np.testing.assert_allclose(z.numpy(), z_shard.numpy(), atol=1e-6, rtol=1e-6)

  def test_rmsnorm(self):
    B, T, embed_size = 4, 10, 20

    norm = nn.RMSNorm(embed_size)
    x = Tensor.rand((B, T, embed_size)).contiguous().realize()
    y = norm(x)

    # for norm layers, the correct way to shard weights is duplication
    norm_sharded = nn.RMSNorm(embed_size)
    norm_sharded.weight.shard_(devices_2, axis=None).realize()

    # if x is being sharded, then all-reduce is involved
    x_sharded = x.shard(devices_2, axis=2).realize()
    y_shard = norm_sharded(x_sharded).realize()
    np.testing.assert_allclose(y.numpy(), y_shard.numpy(), atol=1e-6, rtol=1e-6)

    # if x is being duplicated, then the operations remain inside each GPU
    # which is the common case
    x_sharded = x.shard(devices_2, axis=None).realize()
    y_shard = norm_sharded(x_sharded).realize()
    np.testing.assert_allclose(y.numpy(), y_shard.numpy(), atol=1e-6, rtol=1e-6)

  # NOTE: this is failing on LLVM CI, no idea why. Works locally.
  @unittest.skipIf(CI and Device.DEFAULT in ("CUDA", "NV", "LLVM"), "slow")
  def test_data_parallel_resnet(self):
    import sys, pathlib
    sys.path.append((pathlib.Path(__file__).parent.parent / "extra" / "models").as_posix())
    from resnet import ResNet18

    fake_image = Tensor.rand((2, 3, 224//8, 224//8))
    fake_image_sharded = fake_image.shard(devices_2, axis=0)
    m = ResNet18()
    m.load_from_pretrained()
    real_output = m(fake_image).log_softmax().numpy()
    for p in get_parameters(m): p.shard_(devices_2).realize()
    GlobalCounters.reset()
    shard_output = m(fake_image_sharded).log_softmax().realize()
    assert shard_output.lazydata.lbs[0].shape == (1, 1000)
    assert shard_output.lazydata.lbs[1].shape == (1, 1000)
    shard_output_np = shard_output.numpy()
    np.testing.assert_allclose(real_output, shard_output_np, atol=1e-6, rtol=1e-6)

  @unittest.skipIf(CI and Device.DEFAULT in ("CUDA", "NV", "LLVM"), "slow, and flaky on LLVM")
  def test_data_parallel_resnet_train_step(self):
    import sys, pathlib
    sys.path.append((pathlib.Path(__file__).parent.parent / "extra" / "models").as_posix())
    from resnet import ResNet18
    from tinygrad.nn.optim import LARS

    fake_image = Tensor.rand((2, 3, 224//8, 224//8))
    fake_image_sharded = fake_image.shard(devices_2, axis=0)
    labels = Tensor.randint(2, low=0, high=1000)
    labels_sharded = labels.shard(devices_2, axis=0)

    m = ResNet18()
    optimizer = LARS(get_parameters(m), 0.1)  # set requires_grad for all params

    optimizer.zero_grad()
    m.load_from_pretrained()
    output = m(fake_image).sparse_categorical_crossentropy(labels, label_smoothing=0.1)
    output.backward()
    grad = m.conv1.weight.grad.numpy()

    for p in get_parameters(m): p.shard_(devices_2).realize()
    GlobalCounters.reset()
    optimizer.zero_grad()
    shard_output = m(fake_image_sharded).sparse_categorical_crossentropy(labels_sharded, label_smoothing=0.1)
    assert shard_output.lazydata.axis is None
    shard_output.backward()
    shard_grad = m.conv1.weight.grad.numpy()
    # sometimes there is zeros in these grads... why?
    np.testing.assert_allclose(grad, shard_grad, atol=3e-6, rtol=3e-6)

  def test_multi_tensor_jit_param(self):
    @TinyJit
    def jf(a, b) -> Tensor:
      return (a + b).realize()

    for _ in range(5):
      a = Tensor.ones(256).contiguous().realize()
      b = Tensor.ones(256).contiguous().realize()
      a.shard_(devices_2)
      b.shard_(devices_2)
      c = jf(a, b)
      np.testing.assert_allclose(c.numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
    assert len(jf.jit_cache) > 0

  def test_multi_tensor_jit_body(self):
    @TinyJit
    def jf() -> Tensor:
      a = Tensor.ones(256).contiguous().realize()
      b = Tensor.ones(256).contiguous().realize()
      a.shard_(devices_2)
      b.shard_(devices_2)
      return (a + b).realize()

    for _ in range(5):
      r = jf()
      np.testing.assert_allclose(r.numpy(), np.ones(256)+np.ones(256), atol=1e-4, rtol=1e-5)
    assert len(jf.jit_cache) > 0

  #@unittest.skipIf(CI and Device.DEFAULT=="METAL", "no ICB in CI, creation of graph fails")
  @unittest.skip("test broken")
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
    assert isinstance(jf.jit_cache[4].prg, BufferCopy)
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

  def test_uneven_multiple_zeros(self):
    for data in ([1, 2, 3, 4], [1, 2, 3], [1, 2], [1], []):
      for N in (1, 2, 3, 4):
        devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(N))
        # make sure something is computed on each device
        X = ((Tensor(data).shard(devices, axis=0) + 1).realize() - 1).realize()
        np.testing.assert_equal(X.numpy(), data)

  def test_uneven_shard_with_empty(self):
    N = 4
    X = Tensor.rand(16, 1, 17).contiguous().realize()
    np_x = X.numpy()
    devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(N))

    # test empty shard
    np.testing.assert_equal(X.shard(devices, 0, (2, 2, 12, 0)).numpy(), np_x)

    # test reshape with empty shard
    np.testing.assert_equal(X.shard(devices, 0, (2, 2, 12, 0)).reshape(8, 1, 34).numpy(), np_x.reshape(8, 1, 34))

    # test elementwise with empty shard
    np.testing.assert_equal((X.shard(devices, 0, (2, 2, 12, 0)) + X.shard(devices, 0, (0, 0, 1, 15))).numpy(), np_x + np_x)

  def test_multiple_uneven_shard(self):
    N = 4
    X = Tensor.rand(4, 1, 257).contiguous().realize()
    Y = Tensor.rand(4, 1, 257).contiguous().realize()
    np_x, np_y = X.numpy(), Y.numpy()
    devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(N))
    X.shard_(devices, 2, (2, 38, 47, 170))
    Y.shard_(devices, 2, (34, 53, 51, 119))
    np.testing.assert_equal(X.numpy(), np_x)
    np.testing.assert_equal(Y.numpy(), np_y)
    np.testing.assert_equal((X + Y).numpy(), np_x + np_y)

  def test_bn_ast_on_devices(self):
    t = Tensor.empty((16, 64, 112, 112)).shard(devices_4, axis=0)
    bn = nn.BatchNorm2d(64)
    for p in get_parameters(bn): p.shard_(devices_4).realize()

    out = bn(t)
    scheds = [sched for sched in create_schedule(out.lazydata.lbs) if sched.outputs[0].device in devices_4 and sched.ast.op is not MetaOps.COPY]
    assert set(out.device for sched in scheds for out in sched.outputs) == set(devices_4), "should have ast on each shard device"
    asts = [sched.ast for sched in scheds]
    assert len(asts)
    # test case to show that ast can be different on devices
    # TODO: make ast identical on devices
    #assert len(set(asts)) == 4, len(asts)
    # for i, ast in enumerate(asts):
    #   print(f"{i} {ast}")

  def test_reshape_on_axis(self):
    t0 = Tensor.rand((26, 15, 7)).shard(devices_3, axis=1)

    # test split and rejoin to the right
    t1 = t0.reshape((26, 3, 5, 7))
    t2 = t0.reshape((26, 3, 35))
    t3 = t1.reshape((26, 15, 7))
    t4 = t2.reshape((26, 105,))

    for t in [t0, t1, t2, t3, t4]:
      assert t.lazydata.axis == 1
      np.testing.assert_allclose(t.numpy().flatten(), t0.numpy().flatten())

    # test shape-one axis
    t5 = t4.reshape((26, 1, 105))
    assert t5.lazydata.axis == 2
    np.testing.assert_allclose(t.numpy().flatten(), t5.numpy().flatten())

    # test split and rejoin to the right and reshape to the left
    t5 = t0.reshape((2, 13, 3, 5, 7))
    t6 = t0.reshape((13, 2, 3, 7, 5))
    t7 = t0.reshape((1, 13, 2, 3, 1, 7, 5))
    np.testing.assert_allclose(t5.numpy().flatten(), t0.numpy().flatten())
    assert t5.lazydata.axis == 2
    np.testing.assert_allclose(t6.numpy().flatten(), t0.numpy().flatten())
    assert t6.lazydata.axis == 2
    np.testing.assert_allclose(t7.numpy().flatten(), t0.numpy().flatten())
    assert t7.lazydata.axis == 3

    # test no left join
    with self.assertRaises((AssertionError, ValueError)):
      t0.reshape((26*15,7))

  def test_reshape_on_axis_uneven(self):
    def reshape_helper(t0, t, t_axis):
      np.testing.assert_allclose(t0.reshape(t.shape).numpy(), t.numpy())
      assert t.lazydata.axis == t_axis

    t0 = Tensor.rand((4, 42, 15)).shard(devices_3, axis=1, splits=[14, 7, 21])

    # ok to reshape as long as elements remain on same device
    reshape_helper(t0, t0.reshape(2, 2, 42, 3, 5), 2)
    # split to the right
    reshape_helper(t0, t0.reshape(2, 2, 6, 7, 15), 2)
    # split off and merge to the right
    reshape_helper(t0, t0.reshape(4, 6, 105), 1)
    # really blend the axes together
    reshape_helper(t0, t0.reshape(4, 30, 21), 1)
    # split off 1-shape
    reshape_helper(t0, t0.reshape(4, 1, 42, 15), 2)
    reshape_helper(t0, t0.reshape(4, 6, 1, 7, 15), 1)

    # assert if cannot maintain shard axis without moving items between devices
    with self.assertRaises(AssertionError): t0.reshape(4, 7, 6, 15)
    # assert for degenerate reshape
    with self.assertRaises(AssertionError): t0.reshape(4, 5, 7, 15)
    # assert for cannot maintain axis
    with self.assertRaises(AssertionError): t0.reshape(4, 3, 2, 7, 15)

  def test_mlb_assign_change_axis(self):
    t_none = Tensor.zeros((16, 16)).shard(devices_2).contiguous().realize()
    t_zero = Tensor.ones((16, 16)).shard(devices_2, axis=0)
    with self.assertRaises(AssertionError):
      # don't allow assigns that change axes
      t_none.assign(t_zero)

  def test_dropout_on_shard(self):
    with Tensor.train():
      X = Tensor.ones(256).to(devices_2)
      output = X.dropout(0.5).numpy()
      unique, counts = np.unique(output, return_counts=True)
      assert set(unique) == {0, 2}, unique
      assert 100 < counts[0] < 156, counts[0]

  def test_broadcast_const(self):
    for axis in (None, 0, 1):
      t = Tensor.zeros(16, 16).contiguous().shard(devices_4, axis).realize()
      t = t + 1
      for si in t.schedule():
        ast = si.ast.src[0]
        assert ast.op is UOps.STORE
        assert ast.src[2].arg is BinaryOps.ADD
        assert ast.src[2].src[0].op is UOps.LOAD and ast.src[2].src[0]
        assert ast.src[2].src[1].src[1].op is UOps.CONST and ast.src[2].src[1].src[1].arg == 1
      t = 2 * t
      for si in t.schedule():
        ast = si.ast.src[0]
        assert ast.op is UOps.STORE
        assert ast.src[2].arg is BinaryOps.MUL
        assert ast.src[2].src[0].src[1].op is UOps.CONST and ast.src[2].src[0].src[1].arg == 2
        assert ast.src[2].src[1].op is UOps.LOAD
      t = t + t.full_like(3)
      for si in t.schedule():
        ast = si.ast.src[0]
        assert ast.op is UOps.STORE
        assert ast.src[2].arg is BinaryOps.ADD
        assert ast.src[2].src[0].op is UOps.LOAD
        assert ast.src[2].src[1].src[1].op is UOps.CONST and ast.src[2].src[1].src[1].arg == 3

  def test_shard_memory(self):
    devices = (d0, d1, d2, d3)
    t = Tensor.zeros(16, 16).contiguous()
    t.shard_(devices, axis=0)
    assert all([lb is lb.base and lb.buffer.base.size == 4 * 16 for lb in t.lazydata.lbs])

@unittest.skipIf(CI and Device.DEFAULT in ("GPU", "CUDA", "METAL"), "no GPU CI")
class TestHandleData(unittest.TestCase):
  def test_copied_to_device(self):
    device = (d0, d1, d2, d3)
    t = Tensor([1, 2, 3, 4]).shard(device).realize()
    not_covered = t.to(d5)
    sched = create_schedule([not_covered.lazydata])
    assert len(sched) == 1
    # setup again because create_schedule has side effect
    t = Tensor([1, 2, 3, 4]).shard(device).realize()
    not_covered = t.to(d5)
    assert not_covered.realize().tolist() == [1, 2, 3, 4]

    for d in device:
      t = Tensor([1, 2, 3, 4]).shard(device).realize()
      covered = t.to(d)
      sched = create_schedule([covered.lazydata])
      assert len(sched) == 0
      # setup again because create_schedule has side effect
      t = Tensor([1, 2, 3, 4]).shard(device).realize()
      covered = t.to(d)
      assert covered.realize().tolist() == [1, 2, 3, 4]

@unittest.skipIf(CI and Device.DEFAULT in ("GPU", "CUDA", "METAL"), "no GPU CI")
class TestShrinkMultiTensorShardedAxis(unittest.TestCase):
  # shrink a multitensor on sharded axis
  def test_shrink_bad_args(self):
    t = Tensor.arange(64).reshape(8, 8).contiguous().realize()
    t.shard_([f"{Device.DEFAULT}:{i}" for i in range(4)], axis=0)

    with self.assertRaises(AssertionError):
      # sharded axis shrink on non-device boundry is not allowed
      a = t.shrink(((0, 3), (0, 8)))
    with self.assertRaises(AssertionError):
      # cannot shrink sharded and non-sharded axis at the same time
      a = t.shrink(((0, 2), (2, 4)))

    a = t.shrink(((0, 2), (0, 8)))
    assert a.shape == (2, 8)
    assert a.lazydata.real == [True, False, False, False]

    with self.assertRaises(AssertionError):
      # cannot pad sharded and non-sharded axis at the same time
      p = a.pad(((0, 6), (0, 1)))

    with self.assertRaises(AssertionError):
      # can only pad to whole axis
      p = a.pad(((1, 5), (0, 0)))

    p = a.pad(((0, 6), (0, 0)))
    assert p.shape == (8, 8)
    assert p.lazydata.real == [True, True, True, True]

  @given(strat.sampled_from([dtypes.float, dtypes.int, dtypes.int64, dtypes.int16]))
  def test_ops(self, dtype):
    if not is_dtype_supported(dtype): return
    t = Tensor.arange(64).reshape(8, 8).contiguous().realize()
    t.shard_([f"{Device.DEFAULT}:{i}" for i in range(4)], axis=0)
    for i in range(4):
      print(f"{i=}")
      a = t.shrink(((0+2*i,2+2*i),None))
      b = Tensor(t.numpy()[0+2*i:2+2*i])
      assert a.shape == b.shape == (2, 8)
      assert a.lazydata.real == [i==j for j in range(4)]
      np.testing.assert_allclose(a.numpy(), b.numpy())
      # cast
      np.testing.assert_allclose(a.float().numpy(), b.float().numpy())

      # elementwise
      np.testing.assert_allclose(a.exp().numpy(), b.exp().numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.reciprocal().numpy(), b.reciprocal().numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.pow(-0.5).numpy(), b.pow(-0.5).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose((a+a).numpy(), (b+b).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_equal((a+1).numpy(), (b+1).numpy())
      np.testing.assert_equal((1+a).numpy(), (1+b).numpy())
      np.testing.assert_allclose((a.where(a+a, a)).numpy(), (b.where(b+b, b)).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose((a.where(1, 0)).numpy(), (b.where(1, 0)).numpy(), rtol=1e-7, atol=1e-3)

      # reduce
      np.testing.assert_allclose(a.max().numpy(), b.max().numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.sum().numpy(), b.sum().numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.mean().numpy(), b.mean().numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.max(0).numpy(), b.max(0).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.sum(0).numpy(), b.sum(0).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.mean(0).numpy(), b.mean(0).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.max(1).numpy(), b.max(1).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.sum(1).numpy(), b.sum(1).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.mean(1).numpy(), b.mean(1).numpy(), rtol=1e-7, atol=1e-3)

      # pad it back
      np.testing.assert_allclose(a.pad(((2*i, 2*(4-i-1)), None)).numpy(), b.pad(((2*i, 2*(4-i-1)), None)).numpy(), rtol=1e-7, atol=1e-3)

      # other movement
      np.testing.assert_allclose(a.pad((None, (1, 1))).numpy(), b.pad((None, (1, 1))).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.shrink((None, (1, 3))).numpy(), b.shrink((None, (1, 3))).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.permute((1, 0)).numpy(), b.permute((1, 0)).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.reshape((2, 2, 4)).numpy(), b.reshape((2, 2, 4)).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.reshape((2, 1, 8)).expand((2, 5, 8)).numpy(), b.reshape((2, 1, 8)).expand((2, 5, 8)).numpy(), rtol=1e-7, atol=1e-3)
      np.testing.assert_allclose(a.flip(-1).numpy(), b.flip(-1).numpy(), rtol=1e-7, atol=1e-3)

  def test_uneven(self):
    t = Tensor.arange(24).reshape(3, 8).contiguous().realize()
    t.shard_([f"{Device.DEFAULT}:{i}" for i in range(2)], axis=0)

    a = t.shrink(((0, 2), None))
    b = t.shrink(((2, 3), None))
    na = t.numpy()[0:2]
    nb = t.numpy()[2:3]
    np.testing.assert_equal(a.numpy(), na)
    np.testing.assert_equal(b.numpy(), nb)
    np.testing.assert_equal((a+1).numpy(), na+1)
    np.testing.assert_equal((b+1).numpy(), nb+1)
    np.testing.assert_equal((1+a).numpy(), 1+na)
    np.testing.assert_equal((1+b).numpy(), 1+nb)
    np.testing.assert_equal((a+a).numpy(), na+na)
    np.testing.assert_equal((b+b).numpy(), nb+nb)

  def test_add_two_partitions(self):
    t = Tensor.arange(64).reshape(8, 8).contiguous().realize()
    t.shard_([f"{Device.DEFAULT}:{i}" for i in range(4)], axis=0)

    a = t.shrink(((2, 4), None))
    b = t.shrink(((6, 8), None))
    na = t.numpy()[2:4]
    nb = t.numpy()[6:8]
    np.testing.assert_equal(a.numpy(), na)
    np.testing.assert_equal(b.numpy(), nb)
    with self.assertRaises(AssertionError):
      # cannot add directly
      c = a + b

    c = a.pad(((2, 4), None)) + b.pad(((6, 0), None))
    expected = np.concatenate([np.zeros_like(t.numpy()[0:2]), na, np.zeros_like(t.numpy()[4:6]), nb])
    np.testing.assert_equal(c.numpy(), expected)

  def test_add_different_tensors(self):
    devices = [f"{Device.DEFAULT}:{i}" for i in range(4)]
    x = Tensor.arange(64).reshape(8, 8).contiguous().realize().shard(devices, axis=0)

    to_add = []
    for i in range(len(devices)):
      to_add.append((Tensor.ones(2, 8) * i).shard(devices))

    added:List[Tensor] = []
    for bound, a in zip(x.lazydata.bounds, to_add):
      added.append(x[bound[0]:bound[1]] + a)

    output = added[0].cat(*added[1:])
    expected = np.arange(64).reshape((8,8)) + np.array([[0,0,1,1,2,2,3,3] for _ in range(8)]).T
    np.testing.assert_allclose(output.numpy(), expected)

@unittest.skipIf(CI and Device.DEFAULT in ("GPU", "CUDA", "METAL"), "no GPU CI")
class TestBatchNorm(unittest.TestCase):
  def test_unsynced_backprop_conv_bn(self):
    with Tensor.train():
      from extra.lr_scheduler import OneCycleLR

      convs = [nn.Conv2d(3, 16, 3), nn.Conv2d(3, 16, 3)]
      bns = [nn.BatchNorm2d(16), nn.BatchNorm2d(16)]

      for p in get_parameters(convs + bns):
        p.shard_((d1, d2))
      optim = nn.optim.Adam(get_parameters(convs + bns))
      lr_sched = OneCycleLR(optim, max_lr=0.1, pct_start=0.1, div_factor=100, final_div_factor=0.1, total_steps=10)
      lr_sched.step()

      fake_image = Tensor.rand((8, 3, 32, 32)).shard((d1, d2), axis=0)

      f1 = fake_image.shrink(((0, 4), None, None, None))
      f2 = fake_image.shrink(((4, 8), None, None, None))

      out1 = bns[0](convs[0](f1))
      out2 = bns[1](convs[1](f2))
      out = out1.cat(out2)
      optim.zero_grad()
      out.mean().backward()
      optim.step()
      out.numpy()

  def test_unsynced_backprop_standalone_bn(self):
    from extra.lr_scheduler import OneCycleLR
    GPUS = (d1, d2)

    class BatchNorm:
      def __init__(self, num_features):
        self.bns:List[nn.BatchNorm2d] = []
        for _ in GPUS:
          bn = nn.BatchNorm2d(num_features, track_running_stats=False, eps=1e-12, momentum=0.85, affine=True)
          self.bns.append(bn)

      def __call__(self, x:Tensor):
        bn_ts = []
        for bound, bn in zip(x.lazydata.bounds, self.bns):
          xi = x.shrink((bound, None, None, None))
          bni = bn(xi)
          bn_ts.append(bni)
        return bn_ts[0].cat(*bn_ts[1:])

    with Tensor.train():
      conv = nn.Conv2d(3, 16, 3)
      bn = BatchNorm(16)

      for p in get_parameters([conv, bn]):
        p.shard_(GPUS)
      optim = nn.optim.Adam(get_parameters([conv, bn]))
      lr_sched = OneCycleLR(optim, max_lr=0.1, pct_start=0.1, div_factor=100, final_div_factor=0.1, total_steps=10)
      lr_sched.step()

      fake_image = Tensor.rand((8, 3, 32, 32)).shard(GPUS, axis=0)

      out = bn(conv(fake_image))
      optim.zero_grad()
      out.mean().backward()
      optim.step()

  def test_unsynced_backprop_sync_weights(self):
    from extra.lr_scheduler import OneCycleLR
    from examples.hlb_cifar10 import UnsyncedBatchNorm
    GPUS = (d1, d2)

    with Tensor.train():
      conv = nn.Conv2d(3, 16, 3)
      bn = UnsyncedBatchNorm(16, num_devices=len(GPUS))

      for k, p in get_state_dict([conv, bn]).items():
        if 'running_mean' in k or 'running_var' in k:
          p.shard_(GPUS, axis=0)
        else:
          p.to_(GPUS)
      optim = nn.optim.Adam(get_parameters([conv, bn]))
      lr_sched = OneCycleLR(optim, max_lr=0.1, pct_start=0.1, div_factor=100, final_div_factor=0.1, total_steps=10)
      lr_sched.step()

      fake_image = Tensor.rand((8, 3, 32, 32)).shard(GPUS, axis=0)

      out = bn(conv(fake_image))
      optim.zero_grad()
      out.mean().backward()
      optim.step()

  @given(strat.sampled_from((False, True)))
  def test_batchnorm(self, is_training):
    devices = [f"{Device.DEFAULT}:{i}" for i in range(4)]
    x = Tensor.arange(4096).reshape(8, 8, 8, 8).contiguous().realize().shard(devices, axis=0)

    with Tensor.train(is_training):
      bns = []
      for _ in range(len(devices)):
        bn = nn.BatchNorm2d(8)
        for p in get_parameters(bn):
          p.shard_(devices)
        bn.weight.requires_grad = True
        bn.bias.requires_grad = True
        bns.append(bn)

      bn_ts = []
      for bound, bn in zip(x.lazydata.bounds, bns):
        bni = bn(x[bound[0]:bound[1]])
        bn_ts.append(bni)

      bn_ts[0].cat(*bn_ts[1:]).numpy()

  def test_synced_vs_unsynced_bn(self):
    from examples.hlb_cifar10 import UnsyncedBatchNorm
    from tinygrad.nn import BatchNorm2d
    devices = [f"{Device.DEFAULT}:{i}" for i in range(4)]
    x = Tensor.ones(8, 8, 8, 8).contiguous().realize().shard(devices, axis=0)

    with Tensor.train():
      synced_bn = BatchNorm2d(8)
      unsynced_bn = UnsyncedBatchNorm(8, num_devices=len(devices))

      for p in get_parameters(synced_bn):
        p.shard_(devices)
      for k, p in get_state_dict(unsynced_bn).items():
        if 'running_mean' in k or 'running_var' in k:
          p.shard_(devices, axis=0)
        else:
          p.to_(devices)

      synced_out = synced_bn(x)
      synced_si = list(create_schedule(synced_out.lazydata.lbs))
      unsynced_out = unsynced_bn(x)
      unsynced_si = list(create_schedule(unsynced_out.lazydata.lbs))

    # TODO: test synced / unsynced batchnorm cross device kernel and copies
    assert synced_si
    assert unsynced_si

def helper_test_shard_op(shps, fxn, atol=1e-6, rtol=1e-3):
  for shp in shps:
    single_in = Tensor.randn(shp)
    multi_in  = single_in.shard(devices_2, axis=0)

    single_out = fxn(single_in).numpy()
    multi_out  = fxn(multi_in).numpy()

    try:
      assert single_out.shape == multi_out.shape, f"shape mismatch: single={single_out.shape} | multi={multi_out.shape}"
      assert single_out.dtype == multi_out.dtype, f"dtype mismatch: single={single_out.dtype} | multi={multi_out.dtype}"
      np.testing.assert_allclose(single_out, multi_out, atol=atol, rtol=rtol)
    except Exception as e:
      raise Exception(f"Failed shape {single_out.shape}: {e}")

@unittest.skipIf(CI and Device.DEFAULT in ("GPU", "CUDA", "METAL"), "no GPU CI")
class TestTensorOps(unittest.TestCase):
  def test_interpolate(self):
    helper_test_shard_op([(4,16,16),(4,24,24)], lambda x: Tensor.interpolate(x, (19,19)))

if __name__ == '__main__':
  unittest.main()
