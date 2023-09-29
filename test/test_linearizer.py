import numpy as np
import unittest
from typing import Optional, Any

from tinygrad.codegen.linearizer import Linearizer, UOps
from tinygrad.ops import Compiled, Device, MovementOps, LazyOp
from tinygrad.tensor import Tensor
from tinygrad.jit import CacheCollector
from tinygrad.lazy import _replace_bufferops
from tinygrad.helpers import dtypes

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
    np.testing.assert_allclose(np_c, c.numpy())

  def test_load_dedup(self):
    # for different leaves in the AST, the same loads may occur.

    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled uses linearizer")

    a = Tensor.randn(4).realize()
    # these are of size 3 to avoid float4 coalesce
    r = a[:-1] + a[1:]
    ast = r.lazydata.op
    r = r.realize()  # realize an output buffer
    k = Linearizer(_replace_bufferops(ast)[0], Device[Device.DEFAULT].linearizer_opts)
    k.process()
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
    ast = r.lazydata.op
    r = r.realize()  # realize an output buffer
    k = Linearizer(_replace_bufferops(ast)[0], Device[Device.DEFAULT].linearizer_opts)
    k.process()
    k.upcast()
    k.linearize()
    num_ops = len([uop for uop in k.uops if uop.uop == UOps.ALU])
    assert num_ops <= 1, "more alu uops than needed"

  def test_zero_fold(self):
    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled uses linearizer")

    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = Tensor.stack([a, b])
    ast = r.lazydata.op
    r = r.realize()  # realize an output buffer
    k = Linearizer(_replace_bufferops(ast)[0], Device[Device.DEFAULT].linearizer_opts)
    k.process()
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
    ast = r.lazydata.op
    r = r.realize()  # realize an output buffer
    k = Linearizer(_replace_bufferops(ast)[0], Device[Device.DEFAULT].linearizer_opts)
    k.process()
    k.linearize()
    num_ops = len([uop for uop in k.uops if uop.uop in [UOps.LOAD, UOps.ALU]])
    assert num_ops <= 0, "more load or alu uops than needed"

def helper_linearizer_opt(r:Tensor, opts=[]):
  wanna_output = None
  realized_ast = None
  real_bufs = None

  # HACK to get real ast.
  real_dev_exec_ast = Device[Device.DEFAULT].exec_ast
  def fake_exec_ast(ast, output=None, inputs=None, **kwargs):
    nonlocal realized_ast, real_bufs
    x = real_dev_exec_ast(ast, output, inputs, **kwargs)
    real_bufs = [output.realized] + inputs
    if not(ast.op in MovementOps and ast.src[0].__class__ is not LazyOp and ast.src[0].realized): realized_ast = ast # get last executed
    return x
  Device[Device.DEFAULT].exec_ast = fake_exec_ast
  r = r.realize()  # realize an output buffer
  assert realized_ast is not None
  Device[Device.DEFAULT].exec_ast = real_dev_exec_ast

  def check_opt(x, create_k, to_prg):
    k = create_k()
    k.process()
    k.apply_auto_opt(x)
    prg = to_prg(k)
    real_bufs[0] = real_bufs[0].fromCPU(np.zeros((real_bufs[0].size, ), dtype=real_bufs[0].dtype.np)) # Zero to check that all values are filled
    prg.exec(real_bufs, force_wait=True)
    np.testing.assert_allclose(wanna_output, real_bufs[0].toCPU(), atol=1e-4, rtol=1e-4)

  # Get baseline, which is not optimized at all.
  k = Linearizer(realized_ast, Device[Device.DEFAULT].linearizer_opts)
  k.process()
  prg = Device[Device.DEFAULT].to_program(k)
  prg.exec(real_bufs, force_wait=True)
  wanna_output = real_bufs[0].toCPU().copy()

  # Check correctness of handcoded optimiztions.
  k = Linearizer(realized_ast, Device[Device.DEFAULT].linearizer_opts)
  k.hand_coded_optimizations()
  prg = Device[Device.DEFAULT].to_program(k)
  real_bufs[0] = real_bufs[0].fromCPU(np.zeros((real_bufs[0].size, ), dtype=real_bufs[0].dtype.np)) # Zero to check that all values are filled
  prg.exec(real_bufs, force_wait=True)
  np.testing.assert_allclose(wanna_output, real_bufs[0].toCPU(), atol=1e-4, rtol=1e-4)
  for x in opts: # Check custom transformations if any.
    check_opt(x, lambda: Linearizer(realized_ast, Device[Device.DEFAULT].linearizer_opts), Device[Device.DEFAULT].to_program)

class TestLinearizerOpts(unittest.TestCase):
  def test_local_and_grouped_reduce(self):
    if not isinstance(Device[Device.DEFAULT], Compiled) or not Device[Device.DEFAULT].linearizer_opts.has_local:
      self.skipTest("Only Compiled uses linearizer with locals")

    N = 128
    Tensor.manual_seed(1882)
    a = Tensor.rand(4, 4, N, N)
    b = Tensor.rand(4, 4, N)
    r = (b.sqrt() + ((a+1).sum(axis=3).exp()))
    helper_linearizer_opt(r, [
      [(0, 2, 'L')], [(0, 8, 'L')], [(0, 16, 'L')], # Checking how it works with locals
      [(0, 2, 'G')], [(0, 32, 'G')], [(0, 64, 'G')], # Checking how it works with grouped reduce
      [(0, 2, 'L'), (0, 2, 'G')], [(0, 16, 'L'), (0, 16, 'G')], [(0, 32, 'L'), (0, 2, 'G')], [(0, 2, 'L'), (0, 64, 'G')], # Checking how it works with locals + grouped reduce
      [(0, 2, 'L'), (0, 2, 'G'), (0, 8, 'U'), (0, 4, 'R')], # Checking how it works with locals + grouped reduce + upcasts
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
      [(0, 2, 'U')], [(0, 4, 'U')], [(0, 8, 'U')], # Checking how it works with upcasts
    ])

  def test_full_upcast(self):
    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled uses linearizer")

    Tensor.manual_seed(1772)
    a = Tensor.rand(4)
    b = Tensor.rand(4)
    r = (a+b).sqrt() * ((a+1).exp())
    helper_linearizer_opt(r, [
      [(0, 4, 'U')], # Checking how it works with upcasts
    ])

  def test_matmul(self):
    if not isinstance(Device[Device.DEFAULT], Compiled) or not Device[Device.DEFAULT].linearizer_opts.has_local:
      self.skipTest("Only Compiled uses linearizer with locals")

    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = a@b
    helper_linearizer_opt(r, [
      [(0, 2, 'U')], [(0, 4, 'U'), (1, 4, 'U')], # Checking how it works with upcasts
      [(0, 2, 'L')], [(1, 32, 'L')], [(0, 4, 'L'), (1, 4, 'L')], [(0, 4, 'L'), (1, 32, 'L')], [(0, 16, 'L'), (1, 8, 'L')], # Checking how it works with locals
      [(0, 2, 'G')], [(0, 32, 'G')], [(0, 32, 'G'), (0, 4, 'R')], # Checking how it works with grouped_reduce
      [(0, 2, 'L'), (1, 2, 'L'), (0, 32, 'G')], [(0, 16, 'L'), (0, 32, 'G')], [(0, 16, 'L'), (0, 8, 'L'), (0, 4, 'G')], # Checking how it works with local+grouped_reduce
      [(0, 4, 'L'), (0, 4, 'L'), (0, 16, 'G'), (0, 4, 'R'), (0, 4, 'U'), (1, 2, 'U')], # Checking all together
      [(0, 4, 'L'), (0, 4, 'L'), (0, 16, 'G'), (0, 4, 'R'), (0, 8, 'U')], # Full global upcast + local
    ])

  def test_double_reduce(self):
    if not isinstance(Device[Device.DEFAULT], Compiled) or not Device[Device.DEFAULT].linearizer_opts.has_local:
      self.skipTest("Only Compiled uses linearizer with locals")

    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(8, N, 8, N)
    r = a.sum(axis=(1,3))
    helper_linearizer_opt(r, [
      [(0, 2, 'G')], [(0, 32, 'G')], [(1, 2, 'G')], [(1, 32, 'G')], # Checking how it works with 1 grouped_reduce.
      [(0, 2, 'G'), (1, 2, 'G')], [(0, 16, 'G'), (1, 2, 'G')], [(0, 4, 'G'), (1, 64, 'G')], # Checking how it works with 2 grouped_reduces.
      [(0, 16, 'G'), (1, 2, 'G'), (1, 4, 'R')], [(0, 2, 'G'), (1, 32, 'G'), (1, 4, 'R')], # Checking how it works with 2 grouped_reduces + upcasts.
      [(0, 4, 'L'), (1, 4, 'L'), (0, 8, 'G'), (1, 4, 'G')], [(0, 4, 'L'), (1, 4, 'L'), (0, 2, 'G'), (1, 32, 'G'), (1, 4, 'R')], # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [(0, 2, 'L'), (1, 2, 'L'), (0, 8, 'G'), (1, 4, 'G'), (0, 2, 'U')], [(0, 2, 'L'), (1, 2, 'L'), (0, 8, 'G'), (1, 4, 'G'), (0, 2, 'U'), (0, 4, 'R'), (1, 4, 'R')], # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [(0, 4, 'L'), (1, 4, 'L'), (0, 8, 'G'), (1, 4, 'G'), (0, 2, 'U'), (1, 2, 'U')], # No globals
    ])

class TestFloat4(unittest.TestCase):
  def setUp(self):
    if not Device[Device.DEFAULT].linearizer_opts.supports_float4:
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
    k = Linearizer(s[0])
    k.hand_coded_optimizations()
    k.linearize()

    assert TestFloat4.count_float4(k) == (2, 1)

  def test_float4_multidim(self):
    a = Tensor.rand(2, 8).realize()
    b = Tensor.rand(2, 8).realize()
    c = a + b

    s = c.lazydata.schedule()[0]
    k = Linearizer(s[0])
    k.process()
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
    k = Linearizer(s[0])
    k.hand_coded_optimizations()  # implicit trigger float4 dim
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 1)

  def test_float4_multidim_unaligned_load(self):
    a = Tensor.rand(2, 9).realize().shrink(((0, 2), (1, 9),))
    b = Tensor.rand(2, 9).realize().shrink(((0, 2), (1, 9),))
    c = a + b

    s = c.lazydata.schedule()[0]
    k = Linearizer(s[0])
    k.process()
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
    k = Linearizer(s[0])
    k.process()
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
    k = Linearizer(s[0])
    k.process()
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
    k = Linearizer(s[0])
    k.process()
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
    k = Linearizer(s[0])
    k.process()
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
    k = Linearizer(s[0])
    k.process()
    k.shift_to(0, 4)  # float4 axis
    k.upcast()
    k.linearize()

    assert TestFloat4.count_float4(k) == (1, 1)


if __name__ == '__main__':
  unittest.main()
