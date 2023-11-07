import numpy as np
import unittest, os

from tinygrad.codegen.kernel import Opt, OptOps, tensor_cores
from tinygrad.codegen.linearizer import Linearizer, UOps
from tinygrad.ops import Compiled, Device, LoadOps
from tinygrad.tensor import Tensor
from tinygrad.jit import CacheCollector
from tinygrad.realize import run_schedule
from tinygrad.helpers import dtypes, prod

class TestLinearizer(unittest.TestCase):
  def test_arg_dedup(self):
    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled supports cache")
    a, b = Tensor.randn(4), Tensor.randn(4)
    np_a, np_b = a.numpy(), b.numpy()
    CacheCollector.start()
    c = ((a.shrink(((0, 2),)) - a.shrink(((2, 4),))) - (b.shrink(((0, 2),)) - b.shrink(((2, 4),)))).realize()
    rawbufs = CacheCollector.finish()[0][1]
    assert len(rawbufs) == 3 and set(rawbufs[1:]) == {a.lazydata.realized, b.lazydata.realized}
    np_c = (np_a[:2] - np_a[2:]) - (np_b[:2] - np_b[2:])
    np.testing.assert_allclose(np_c, c.numpy(), atol=1e-4, rtol=1e-4)

  def test_load_dedup(self):
    # for different leaves in the AST, the same loads may occur.

    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled uses linearizer")

    a = Tensor.randn(4).realize()
    # these are of size 3 to avoid float4 coalesce
    r = a[:-1] + a[1:]

    k = Linearizer(r.lazydata.schedule()[-1].ast)
    k.upcast()
    k.linearize()
    num_loads = len([uop for uop in k.uops if uop.uop == UOps.LOAD])
    assert num_loads <= 4, "more load uops than needed"
    assert num_loads >= 4, "unexpected number of uops, maybe this test needs updating?"

  def test_upcast_cse(self):
    # when upcasting, within a subtree, there may be common expressions.

    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled uses linearizer")

    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = a.expand([2]) + b.expand([2])

    k = Linearizer(r.lazydata.schedule()[-1].ast)
    k.upcast()
    k.linearize()
    num_ops = len([uop for uop in k.uops if uop.uop == UOps.ALU])
    assert num_ops <= 1, "more alu uops than needed"

  def test_zero_fold(self):
    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled uses linearizer")

    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = Tensor.stack([a, b])

    k = Linearizer(r.lazydata.schedule()[-1].ast)
    k.upcast()
    k.linearize()
    num_ops = len([uop for uop in k.uops if uop.uop == UOps.ALU])
    assert num_ops == 0, "more alu uops than needed"

  @unittest.skip("constant folding not supported yet")
  def test_constant_fold(self):
    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled uses linearizer")

    a, b = Tensor(2), Tensor(3)
    r = a * b

    k = Linearizer(r.lazydata.schedule()[-1][0])
    k.linearize()
    num_ops = len([uop for uop in k.uops if uop.uop in [UOps.LOAD, UOps.ALU]])
    assert num_ops <= 0, "more load or alu uops than needed"

  def test_tensor_cores(self):
    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled uses linearizer")
    if Device.DEFAULT not in tensor_cores:
      self.skipTest("No tensor cores for device")

    for tc in tensor_cores[Device.DEFAULT]:
      if tc.arch is not None and tc.arch != os.uname().machine: continue
      a, b = Tensor.rand(tc.dims[0], tc.dims[2], dtype=tc.dtype_in), Tensor.rand(tc.dims[2], tc.dims[1], dtype=tc.dtype_in)
      np_a, np_b = a.numpy(), b.numpy()
      if tc.dtype_out != tc.dtype_in:
        r = (a.reshape(tc.dims[0], 1, tc.dims[2]) * b.permute(1,0).reshape(1, tc.dims[1], tc.dims[2])).cast(tc.dtype_out).sum(axis=2)
      else:
        r = a @ b
      realized_ast, _ = helper_realized_ast(r)
      k = Linearizer(realized_ast)
      k.apply_tensor_cores(1)
      k.linearize()
      assert len([uop for uop in k.uops if uop.uop == UOps.WMMA]) == 1, "tensor core not triggered"
      np_c = np_a @ np_b
      np.testing.assert_allclose(np_c, r.numpy(), atol=5e-3, rtol=1e-4)

  def test_limit_dims_to_max_5d_global(self):
    t = Tensor.rand(3, 4, 5, 6, 7).pad(((1, 1), (1, 1), (1, 1), (1, 1), (1, 1))) + 1
    sched = [si for si in t.lazydata.schedule() if si.ast.op not in LoadOps]
    assert len(sched) == 1
    lin = Linearizer(sched[0].ast)
    assert lin.full_shape[:lin.global_dims] == (5, 6, 7, 8, 9)
    lin.limit_dims_to_max(global_max=[16, 16, 16], local_max=[16, 16, 16])

def helper_realized_ast(r:Tensor):
  s = r.lazydata.schedule()
  run_schedule(s[:-1])  # run all kernels except the last one
  # now all input LazyBuffers buffers in s[-1] should be realized
  output_buffer = Device[s[-1].out.device].buffer(prod((s if isinstance(s, int) else s.max for s in s[-1].out.shape)), s[-1].out.dtype, **s[-1].out._device_extra_args())  # allocate an output buffer
  return s[-1].ast, [output_buffer] + [l.realized for l in s[-1].inputs]

class TestFloat4(unittest.TestCase):
  def setUp(self):
    if not isinstance(Device[Device.DEFAULT], Compiled) or not Device[Device.DEFAULT].linearizer_opts.supports_float4:
      self.skipTest("Device does not support float4")

  @staticmethod
  def count_float4(k):
    return (len([uop for uop in k.uops if uop.uop == UOps.LOAD and uop.dtype == dtypes._float4]),
            len([uop for uop in k.uops if uop.uop == UOps.STORE and len(uop.vin) == 3 and uop.vin[2].dtype == dtypes._float4]))

  # TODO: express opts below as auto opts

  def test_float4_basic(self):
    a = Tensor.rand(2, 8).realize()
    b = Tensor.rand(2, 8).realize()
    c = a + b

    s = c.lazydata.schedule()[0]
    k = Linearizer(s.ast)
    k.hand_coded_optimizations()
    k.linearize()

    assert TestFloat4.count_float4(k) == (2, 1)

  def test_float4_multidim(self):
    a = Tensor.rand(2, 8).realize()
    b = Tensor.rand(2, 8).realize()
    c = a + b

    s = c.lazydata.schedule()[0]
    k = Linearizer(s.ast)
    k.shift_to(0, 4)  # float4 dimension
    k.shift_to(0, 2, insert_before=k.shape_len-1)
    k.upcast()
    k.upcast()
    k.local_dims += 1
    k.linearize()

    assert TestFloat4.count_float4(k) == (4, 2)

  def test_float4_unaligned_load(self):
    a = Tensor.rand(9).realize().shrink(((1, 9),))
    b = Tensor.rand(9).realize().shrink(((1, 9),))
    c = a + b

    s = c.lazydata.schedule()[0]
    k = Linearizer(s.ast)
    k.hand_coded_optimizations()  # implicit trigger float4 dim
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 1)

  def test_float4_multidim_unaligned_load(self):
    a = Tensor.rand(2, 9).realize().shrink(((0, 2), (1, 9),))
    b = Tensor.rand(2, 9).realize().shrink(((0, 2), (1, 9),))
    c = a + b

    s = c.lazydata.schedule()[0]
    k = Linearizer(s.ast)
    k.shift_to(len(k.full_unupcasted_shape)-1, 4)  # manual trigger float4 dim
    k.upcast()
    k.shift_to(len(k.full_unupcasted_shape)-1, 2, insert_before=k.shape_len-1)
    k.upcast()
    k.local_dims += 1
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 2)

  def test_float4_sometimes_unaligned(self):
    a = Tensor.rand(1, 1, 8).realize()
    b = Tensor.rand(1, 1, 5).realize().shrink(((0, 1), (0, 1), (1, 5)))
    c = a.conv2d(b)
    # only the first and last conv dot products are aligned in a, and b is never aligned, so no
    # float4 should be emitted (the reduce axis of size 4 is the float4 axis here)

    s = c.lazydata.schedule()[0]
    k = Linearizer(s.ast)
    k.upcast()
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 0)

  def test_float4_multidim_sometimes_unaligned(self):
    a = Tensor.rand(1, 1, 7).realize()
    b = Tensor.rand(1, 1, 5).realize().shrink(((0, 1), (0, 1), (1, 5)))
    c = a.conv2d(b)
    # the first conv dot product is aligned in a. If we upcast the output and reduce
    # dimension, then we could do float4 for only that one set of loads, but we currently
    # don't.

    s = c.lazydata.schedule()[0]
    k = Linearizer(s.ast)
    k.upcast()
    k.upcast()
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 1)

  def test_float4_noncontiguous(self):
    a = Tensor.rand(4, 2).realize()
    b = Tensor.rand(4, 2).realize()
    c = a + b

    # we will upcast the top axis of sz 4. they should not be coalesced into float4,
    # since the top axis is not contiguous.

    s = c.lazydata.schedule()[0]
    k = Linearizer(s.ast)
    k.shift_to(0, 4, top=True)  # top axes are float4 axes
    k.upcast()
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 0)

  def test_float4_expand(self):
    a = Tensor.rand(9).realize().shrink(((1, 9),))
    b = Tensor.rand(2).realize().reshape((2, 1)).expand((2,4)).reshape((8,))
    c = a + b

    # we will upcast the top axis of sz 4. they should not be coalesced into float4,
    # since the top axis is not contiguous.

    s = c.lazydata.schedule()[0]
    k = Linearizer(s.ast)
    k.shift_to(0, 4)  # float4 axis
    k.upcast()
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 1)

  def test_float4_heterogeneous(self):
    a = Tensor.rand(8).realize()
    b = Tensor.rand(9).realize().shrink(((1, 9),))
    c = a + b

    # should float4 b but not a

    s = c.lazydata.schedule()[0]
    k = Linearizer(s.ast)
    k.shift_to(0, 4)  # float4 axis
    k.upcast()
    k.linearize()

    assert TestFloat4.count_float4(k) == (1, 1)

class TestHandCodedOpts(unittest.TestCase):
  def setUp(self):
    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Device does not use linearizer")

  def test_masked_upcast(self):
    layer_1 = Tensor.cat(*[Tensor.rand(5) for _ in range(4)])
    layer_2 = Tensor.cat(layer_1.unsqueeze(0), Tensor.rand(6, 20))

    s = layer_2.lazydata.schedule()[-1]
    k = Linearizer(s.ast)
    k.hand_coded_optimizations()
    assert len(k.bufs) == 6  # make sure all ops are done in one kernel
    # masked upcast should upcast masked axis of size 7
    # masked upcast should not upcast large (20) last axis
    # float4/other hcopt shouldn't upcast last axis, since we already have 7 upcast, and the last axis is not very contiguous
    assert k.upcasted == 1 and k.full_shape[-1] == 7

  def test_masked_upcast_wino(self):
    monster = Tensor.stack([Tensor.stack([Tensor.rand(16) for _ in range(6)]) for _ in range(6)])

    s = monster.lazydata.schedule()[-1]
    k = Linearizer(s.ast)
    k.hand_coded_optimizations()
    assert len(k.bufs) == 37  # make sure all ops are done in one kernel
    # should upcast the two Tensor.stacks
    assert k.upcasted >= 2 and k.full_shape[k.shape_len-k.upcasted:k.shape_len].count(6) == 2

  def test_masked_upcast_wino_full(self):
    old_wino = Tensor.wino
    Tensor.wino = True
    x,w = Tensor.rand(1,4,9,9, requires_grad=True).realize(), Tensor.rand(4,4,3,3, requires_grad=True).realize()
    out = Tensor.conv2d(x,w, padding=1)
    upcasts = []
    # collect upcasts of tile transform kernels
    for i, si in enumerate(out.lazydata.schedule()):
      k = Linearizer(si.ast)
      k.hand_coded_optimizations()
      if k.reduceop is not None: continue  # not a tile transform kernel (there is a gemm reduce kernel)
      if len(k.bufs) < 100: continue  # not a tile transform kernel (there's a permute kernel at the end)
      upcasts.append(tuple(k.full_shape[k.shape_len - k.upcasted:k.shape_len]))
    assert len(upcasts) == 3  # 3 transformation matrices
    assert upcasts.count((6, 6)) == 2 and upcasts.count((4, 4)) == 1

    out.mean().backward()
    for si in x.grad.lazydata.schedule() + w.grad.lazydata.schedule():
      k = Linearizer(si.ast)
      k.hand_coded_optimizations()
      k.linearize()
      if len(k.bufs) < 20: continue  # not a tile transform kernel
      # heuristic number to make sure that at least some upcasts but not too many upcasts are being done
      assert 6 <= prod(k.full_shape[k.shape_len - k.upcasted:k.shape_len]) <= 49

    Tensor.wino = old_wino

  def test_masked_upcast_many(self):
    layer_1 = Tensor.cat(Tensor.rand(3, 4), Tensor.rand(4, 4))
    layer_2 = Tensor.cat(layer_1.unsqueeze(0), Tensor.rand(6, 7, 4))
    layer_3 = Tensor.cat(layer_2.unsqueeze(0), Tensor.rand(6, 7, 7, 4))

    s = layer_3.lazydata.schedule()[-1]
    k = Linearizer(s.ast)
    k.hand_coded_optimizations()
    assert len(k.bufs) == 5  # make sure all ops are done in one kernel
    # check that we don't do too many upcasts
    assert prod(k.full_shape[k.shape_len-k.upcasted:k.shape_len]) <= 49

def helper_linearizer_opt(r:Tensor, opts=[], apply_tc=False):
  wanna_output = None
  realized_ast, real_bufs = helper_realized_ast(r)

  def check_opt(opts, create_k, to_prg):
    k = create_k()
    if apply_tc:
      k.apply_tensor_cores(1, opts)
    else:
      for opt in opts:
        k.apply_opt(opt)
    prg = to_prg(k)
    real_bufs[0] = real_bufs[0].fromCPU(np.zeros((real_bufs[0].size, ), dtype=real_bufs[0].dtype.np)) # Zero to check that all values are filled
    prg.exec(real_bufs, force_wait=True)
    np.testing.assert_allclose(wanna_output, real_bufs[0].toCPU(), atol=1e-4, rtol=1e-4)

  # Get baseline, which is not optimized at all.
  k = Linearizer(realized_ast)
  prg = Device[Device.DEFAULT].to_program(k)
  prg.exec(real_bufs, force_wait=True)
  wanna_output = real_bufs[0].toCPU().copy()

  # Check correctness of handcoded optimiztions.
  k = Linearizer(realized_ast)
  k.hand_coded_optimizations()
  prg = Device[Device.DEFAULT].to_program(k)
  real_bufs[0] = real_bufs[0].fromCPU(np.zeros((real_bufs[0].size, ), dtype=real_bufs[0].dtype.np)) # Zero to check that all values are filled
  prg.exec(real_bufs, force_wait=True)
  np.testing.assert_allclose(wanna_output, real_bufs[0].toCPU(), atol=1e-4, rtol=1e-4)
  for x in opts: # Check custom transformations if any.
    check_opt(x, lambda: Linearizer(realized_ast), Device[Device.DEFAULT].to_program)

class TestLinearizerOpts(unittest.TestCase):
  def test_local_and_grouped_reduce(self):
    if not isinstance(Device[Device.DEFAULT], Compiled) or not Device[Device.DEFAULT].linearizer_opts.has_local or not Device[Device.DEFAULT].linearizer_opts.has_shared:
      self.skipTest("Only Compiled uses linearizer with locals and shared")

    N = 128
    Tensor.manual_seed(1882)
    a = Tensor.rand(4, 4, N, N)
    b = Tensor.rand(4, 4, N)
    r = (b.sqrt() + ((a+1).sum(axis=3).exp()))
    helper_linearizer_opt(r, [
      [Opt(OptOps.LOCAL, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 8)],
      [Opt(OptOps.LOCAL, 0, 16)], # Checking how it works with locals
      [Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 0, 64)], # Checking how it works with grouped reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.GROUPTOP, 0, 16)],
      [Opt(OptOps.LOCAL, 0, 32), Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 64)], # Checking how it works with locals + grouped reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.UPCAST, 0, 8), Opt(OptOps.UNROLL, 1, 4)], # Checking how it works with locals + grouped reduce + upcasts
    ])

  def test_upcasts(self):
    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled uses linearizer")

    N = 16
    Tensor.manual_seed(1772)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = (a+b).sqrt() * ((a+1).exp())
    helper_linearizer_opt(r, [
      [Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 4)],
      [Opt(OptOps.UPCAST, 0, 8)], # Checking how it works with upcasts
    ])

  def test_full_upcast(self):
    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled uses linearizer")

    Tensor.manual_seed(1772)
    a = Tensor.rand(4)
    b = Tensor.rand(4)
    r = (a+b).sqrt() * ((a+1).exp())
    helper_linearizer_opt(r, [
      [Opt(OptOps.UPCAST, 0, 4)], # Checking how it works with upcasts
    ])

  def test_matmul(self):
    if not isinstance(Device[Device.DEFAULT], Compiled) or not Device[Device.DEFAULT].linearizer_opts.has_local or not Device[Device.DEFAULT].linearizer_opts.has_shared:
      self.skipTest("Only Compiled uses linearizer with locals and shared")

    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = a@b
    helper_linearizer_opt(r, [
      [Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4)], # Checking how it works with upcasts
      [Opt(OptOps.LOCAL, 0, 2)],
      [Opt(OptOps.LOCAL, 1, 32)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 32)],
      [Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.LOCAL, 1, 8)], # Checking how it works with locals
      [Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 0, 32), Opt(OptOps.UNROLL, 0, 4)], # Checking how it works with grouped_reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.GROUPTOP, 0, 4)], # Checking how it works with local+grouped_reduce
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 2)], # Checking all together
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 8)], # Full global upcast + local
    ])

  def test_double_reduce(self):
    if not isinstance(Device[Device.DEFAULT], Compiled) or not Device[Device.DEFAULT].linearizer_opts.has_local or not Device[Device.DEFAULT].linearizer_opts.has_shared:
      self.skipTest("Only Compiled uses linearizer with locals and shared")

    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(8, N, 8, N)
    r = a.sum(axis=(1,3))
    helper_linearizer_opt(r, [
      # openCL / GPU=1 is 256 max threads
      [Opt(OptOps.GROUPTOP, 0, 2)], [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 1, 2)], [Opt(OptOps.GROUPTOP, 1, 32)], # Checking how it works with 1 grouped_reduce.
      [Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 2)],
      [Opt(OptOps.GROUPTOP, 0, 16), Opt(OptOps.GROUPTOP, 1, 2)],
      [Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 64)], # Checking how it works with 2 grouped_reduces.
      [Opt(OptOps.GROUPTOP, 0, 16), Opt(OptOps.GROUPTOP, 1, 2), Opt(OptOps.UNROLL, 0, 4)],
      [Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 32), Opt(OptOps.UNROLL, 2, 4)], # Checking how it works with 2 grouped_reduces + upcasts.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 4)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 32), Opt(OptOps.UNROLL, 1, 4)], # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UNROLL, 1, 4)], # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.UPCAST, 0, 2)], # No globals
    ])

  def test_tensor_core_opts(self):
    if not isinstance(Device[Device.DEFAULT], Compiled) or not Device[Device.DEFAULT].linearizer_opts.has_local:
      self.skipTest("Only Compiled uses linearizer with locals")
    if Device.DEFAULT not in tensor_cores:
      self.skipTest("No tensor cores for device")

    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = a@b
    helper_linearizer_opt(r, [
      [Opt(OptOps.UPCAST, 0, 4)],
      [Opt(OptOps.UPCAST, 1, 4)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4)], # check upcasts
      [Opt(OptOps.UNROLL, 0, 2)], # check last unroll
      [Opt(OptOps.LASTLOCAL, 0, 4)], # check last local
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 2)], # check combo of last unroll and last local
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.LASTLOCAL, 0, 2)],
      # [Opt(OptOps.GROUP, 0, 2)] # doesn't work because group_for_reduce dims become early locals (conflicting with TC)
    ], apply_tc=True)


if __name__ == '__main__':
  unittest.main()
