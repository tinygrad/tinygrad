import numpy as np
import unittest, os

from tinygrad.codegen.kernel import tensor_cores
from tinygrad.codegen.linearizer import Linearizer, UOps
from tinygrad.ops import Compiled, Device, MovementOps, LazyOp
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


if __name__ == '__main__':
  unittest.main()
