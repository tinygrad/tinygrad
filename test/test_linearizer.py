import numpy as np
import unittest

from tinygrad.codegen.linearizer import Linearizer, UOps
from tinygrad.ops import Compiled, Device, MovementOps, LazyOp
from tinygrad.tensor import Tensor
from tinygrad.jit import CacheCollector

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
    k = Linearizer(ast, r.lazydata, Device[Device.DEFAULT].linearizer_opts)
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
    k = Linearizer(ast, r.lazydata, Device[Device.DEFAULT].linearizer_opts)
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
    k = Linearizer(ast, r.lazydata, Device[Device.DEFAULT].linearizer_opts)
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
    k = Linearizer(ast, r.lazydata, Device[Device.DEFAULT].linearizer_opts)
    k.process()
    k.linearize()
    num_ops = len([uop for uop in k.uops if uop.uop in [UOps.LOAD, UOps.ALU]])
    assert num_ops <= 0, "more load or alu uops than needed"

def helper_linearizer_opt(r:Tensor, opts=[]):
  wanna_output = None
  realized_ast = None

  # HACK to get real ast.
  real_dev_exec_ast = Device[Device.DEFAULT].exec_ast
  def fake_exec_ast(ast, output=None, **kwargs):
    nonlocal realized_ast
    x = real_dev_exec_ast(ast, output, **kwargs)
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
    k.bufs[0].realized = k.bufs[0].realized.fromCPU(np.zeros(k.bufs[0].shape, dtype=k.bufs[0].dtype.np)) # Zero to check that all values are filled
    prg.exec(k.bufs, force_wait=True)
    np.testing.assert_allclose(wanna_output, k.bufs[0].toCPU(), atol=1e-4, rtol=1e-4)

  # Get baseline, which is not optimized at all.
  k = Linearizer(realized_ast, r.lazydata, Device[Device.DEFAULT].linearizer_opts)
  k.process()
  prg = Device[Device.DEFAULT].to_program(k)
  prg.exec(k.bufs, force_wait=True)
  wanna_output = k.bufs[0].toCPU().copy()

  # Check correctness of handcoded optimiztions.
  k = Linearizer(realized_ast, r.lazydata, Device[Device.DEFAULT].linearizer_opts)
  k.hand_coded_optimizations()
  prg = Device[Device.DEFAULT].to_program(k)
  k.bufs[0].realized = k.bufs[0].realized.fromCPU(np.zeros(k.bufs[0].shape, dtype=k.bufs[0].dtype.np)) # Zero to check that all values are filled
  prg.exec(k.bufs, force_wait=True)
  np.testing.assert_allclose(wanna_output, k.bufs[0].toCPU(), atol=1e-4, rtol=1e-4)
  for x in opts: # Check custom transformations if any.
    check_opt(x, lambda: Linearizer(realized_ast, r.lazydata, Device[Device.DEFAULT].linearizer_opts), Device[Device.DEFAULT].to_program)

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

if __name__ == '__main__':
  unittest.main()
