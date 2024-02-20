import numpy as np
import unittest

from tinygrad.codegen.kernel import Opt, OptOps, tensor_cores
from tinygrad.codegen.linearizer import Linearizer, UOp, UOps, expand_node
from tinygrad.device import Compiled, Device, Buffer
from tinygrad.ops import BinaryOps, BufferOps, MemBuffer, ConstBuffer, LazyOp, LoadOps, TernaryOps
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import MulNode, Variable, NumNode, Node
from tinygrad.tensor import Tensor
from tinygrad.features.jit import CacheCollector
from tinygrad.realize import create_schedule, run_schedule
from tinygrad.helpers import prod, Context
from tinygrad.dtype import DType, dtypes

@unittest.skipIf(not isinstance(Device[Device.DEFAULT], Compiled), "linearizer is only for compiled backends")
class TestLinearizer(unittest.TestCase):
  def test_arg_dedup(self):
    a, b = Tensor.randn(4), Tensor.randn(4)
    np_a, np_b = a.numpy(), b.numpy()
    CacheCollector.start()
    c = ((a.shrink(((0, 2),)) - a.shrink(((2, 4),))) - (b.shrink(((0, 2),)) - b.shrink(((2, 4),)))).realize()
    rawbufs = CacheCollector.finish()[0].rawbufs
    assert len(rawbufs) == 3 and set(rawbufs[1:]) == {a.lazydata.base.realized, b.lazydata.base.realized}
    np_c = (np_a[:2] - np_a[2:]) - (np_b[:2] - np_b[2:])
    np.testing.assert_allclose(np_c, c.numpy(), atol=1e-4, rtol=1e-4)

  def test_load_dedup(self):
    # for different leaves in the AST, the same loads may occur.

    a = Tensor.randn(4).realize()
    # these are of size 3 to avoid float4 coalesce
    r = a[:-1] + a[1:]

    k = Linearizer(create_schedule([r.lazydata])[-1].ast)
    k.upcast()
    k.linearize()
    num_loads = len([uop for uop in k.uops if uop.uop == UOps.LOAD])
    assert num_loads <= 4, "more load uops than needed"
    assert num_loads >= 4, "unexpected number of uops, maybe this test needs updating?"

  def test_load_cache_const_bufs(self):
    # make sure const buffers are differentiated from local and mem buffers
    ST, DT = ShapeTracker(views=(View(shape=((1,)), strides=(0, 0), offset=0, mask=None, contiguous=False),)), dtypes.int
    VAL = LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=2, dtype=DT, st=ST))

    # data1[0] + VAL
    a = LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=DT, st=ST)), VAL))
    # (literal const 1) + VAL
    b = LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1, dtype=DT, st=ST)), VAL))

    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=BinaryOps.ADD, src=(a,b)),), arg=MemBuffer(idx=0, dtype=DT, st=ST))
    lin = Linearizer(ast)
    lin.linearize()

    a_bufs = [u.uop for u in lin.uops[-2].vin[0].vin]
    b_bufs = [u.uop for u in lin.uops[-2].vin[1].vin]

    assert a_bufs == [UOps.LOAD, UOps.CONST]
    assert b_bufs == [UOps.CONST, UOps.CONST]

  def test_upcast_cse(self):
    # when upcasting, within a subtree, there may be common expressions.

    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = a.expand([2]) + b.expand([2])

    k = Linearizer(create_schedule([r.lazydata])[-1].ast)
    k.upcast()
    k.linearize()
    num_ops = len([uop for uop in k.uops if uop.uop == UOps.ALU])
    assert num_ops <= 1, "more alu uops than needed"

  def test_zero_fold(self):
    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = Tensor.stack([a, b])

    k = Linearizer(create_schedule([r.lazydata])[-1].ast)
    k.upcast()
    k.linearize()
    num_ops = len([uop for uop in k.uops if uop.uop == UOps.ALU])
    assert num_ops == 0, "more alu uops than needed"

  @unittest.skip("constant folding not supported yet")
  def test_constant_fold(self):
    a, b = Tensor(2), Tensor(3)
    r = a * b

    k = Linearizer(create_schedule([r.lazydata])[-1].ast)
    k.linearize()
    num_ops = len([uop for uop in k.uops if uop.uop in [UOps.LOAD, UOps.ALU]])
    assert num_ops <= 0, "more load or alu uops than needed"

  def test_sum_acc_dtype(self):
    for tensor_dtype, acc_dtype in (
      (dtypes.bool, dtypes.int), (dtypes.int16, dtypes.int), (dtypes.float16, dtypes.float), (dtypes.bfloat16, dtypes.float)):
      a = Tensor([1, 2, 3], dtype=tensor_dtype).sum()
      k = Linearizer(create_schedule([a.lazydata])[-1].ast)
      k.linearize()
      local = [uop for uop in k.uops if uop.uop == UOps.DEFINE_ACC]
      assert local[0].dtype == acc_dtype

  def test_arg_acc_dtype(self):
    def helper_arg_acc_dtype(c: Tensor, expected_dtype:DType):
      k = Linearizer(create_schedule([c.lazydata])[-1].ast)
      k.linearize()
      local = [uop for uop in k.uops if uop.uop == UOps.DEFINE_ACC]
      assert local[0].dtype == expected_dtype

    tests = (
      (dtypes.float16, None, dtypes.float),
      (dtypes.bfloat16, None, dtypes.float),
      (dtypes.float, None, dtypes.float),
      (dtypes.float16, dtypes.float16, dtypes.float16),
      (dtypes.bfloat16, dtypes.bfloat16, dtypes.bfloat16),
      (dtypes.float, dtypes.float16, dtypes.float16),
    )
    for tensor_dtype, acc_dtype, expected_dtype in tests:
      a, b = Tensor.rand(8, 8, dtype=tensor_dtype), Tensor.rand(8, 8, dtype=tensor_dtype)
      helper_arg_acc_dtype(a.sum(acc_dtype=acc_dtype), expected_dtype)
      helper_arg_acc_dtype(a.matmul(b, acc_dtype=acc_dtype), expected_dtype)
      d, w = Tensor.rand(4, 8, 8, 8, dtype=tensor_dtype), Tensor.rand(8, 8, 2, 2, dtype=tensor_dtype)
      helper_arg_acc_dtype(d.conv2d(w, acc_dtype=acc_dtype), expected_dtype)

  def test_tensor_cores(self):
    if not Device[Device.DEFAULT].compiler.linearizer_opts.has_tensor_cores:
      self.skipTest("device doesn't have tensor cores")
    for tc in tensor_cores[Device[Device.DEFAULT].compiler.linearizer_opts.device]:
      a, b = Tensor.rand(tc.dims[1], tc.dims[2], dtype=tc.dtype_in), Tensor.rand(tc.dims[2], tc.dims[0], dtype=tc.dtype_in)
      np_a, np_b = a.numpy(), b.numpy()
      r = a.matmul(b, acc_dtype=tc.dtype_out)
      realized_ast, _ = helper_realized_ast(r)
      k = Linearizer(realized_ast)
      k.apply_tensor_cores(1)
      k.linearize()
      assert len([uop for uop in k.uops if uop.uop == UOps.WMMA]) == 1, "tensor core not triggered"
      np_c = np_a @ np_b
      (tc_atol, tc_rtol) = (1e-2, 1e-3) if tc.dtype_out == dtypes.half else (5e-3, 1e-4)
      np.testing.assert_allclose(np_c, r.numpy(), atol=tc_atol, rtol=tc_rtol)

  def test_limit_dims_to_max_5d_global(self):
    t = Tensor.rand(3, 4, 5, 6, 7).pad(((1, 1), (1, 1), (1, 1), (1, 1), (1, 1))) + 1
    sched = [si for si in create_schedule([t.lazydata]) if si.ast.op not in LoadOps]
    assert len(sched) == 1
    lin = Linearizer(sched[0].ast)
    assert lin.full_shape[:lin.global_dims] == (5, 6, 7, 8, 9)
    lin.limit_dims_to_max(global_max=[16, 16, 16], local_max=[16, 16, 16])

  def test_sum_collapse(self):
    t = Tensor.ones(256,256).sum()
    sched = [si for si in create_schedule([t.lazydata]) if si.ast.op not in LoadOps]
    assert len(sched) == 1
    lin = Linearizer(sched[0].ast)
    assert not any(u.uop == UOps.LOOP for u in lin.linearize().uops), "found loop in sum collapse"

  def test_simplify_uop(self):
    def helper_test_simplify(uop, dtype, vin, arg=None):
      ast = LazyOp(BufferOps.CONST, (),
                   ConstBuffer(42, dtypes.float, ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),))))
      ast = LazyOp(BufferOps.STORE, (ast,),
                   MemBuffer(0, dtypes.float, ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),))))
      lin = Linearizer(ast=ast) # this is a dummy ast

      lin.uops = []
      return lin.uop(uop, dtype, vin, arg, cachable=False)

    c0 = UOp(UOps.CONST, dtypes.float, vin=(), arg=0.0)
    assert helper_test_simplify(UOps.ALU, dtypes.float, vin=(UOp(UOps.CONST, dtypes.bool, vin=(), arg=True), c0, c0), arg=TernaryOps.WHERE) == c0

    c0 = UOp(UOps.CONST, dtypes.float, vin=(), arg=0.0)
    c1 = UOp(UOps.CONST, dtypes.float, vin=(), arg=1.0)
    assert helper_test_simplify(UOps.ALU, dtypes.float, vin=(UOp(UOps.CONST, dtypes.bool, vin=(), arg=True), c0, c1),
                                arg=TernaryOps.WHERE).uop == UOps.ALU

def helper_realized_ast(r:Tensor):
  s = create_schedule([r.lazydata])
  run_schedule(s[:-1])  # run all kernels except the last one
  # now all input LazyBuffers buffers in s[-1] should be realized
  # allocate an output buffer
  output_buffer = Buffer(s[-1].out.device, prod((s if isinstance(s, int) else s.max for s in s[-1].out.shape)), s[-1].out.dtype)
  return s[-1].ast, [output_buffer] + [l.realized for l in s[-1].inputs]

@unittest.skipUnless(isinstance(Device[Device.DEFAULT], Compiled) and Device[Device.DEFAULT].compiler.linearizer_opts.supports_float4,
                     "need Compiled backends that support float4")
class TestFloat4(unittest.TestCase):
  @staticmethod
  def count_float4(k):
    return (len([uop for uop in k.uops if uop.uop == UOps.LOAD and uop.dtype == dtypes.float.vec(4)]),
            len([uop for uop in k.uops if uop.uop == UOps.STORE and len(uop.vin) == 3 and uop.vin[2].dtype == dtypes.float.vec(4)]))

  # TODO: express opts below as auto opts

  def test_float4_basic(self):
    a = Tensor.rand(2, 8).realize()
    b = Tensor.rand(2, 8).realize()
    c = a + b

    s = create_schedule([c.lazydata])[0]
    k = Linearizer(s.ast)
    k.hand_coded_optimizations()
    k.linearize()

    assert TestFloat4.count_float4(k) == (2, 1)

  def test_float4_multidim(self):
    a = Tensor.rand(2, 8).realize()
    b = Tensor.rand(2, 8).realize()
    c = a + b

    s = create_schedule([c.lazydata])[0]
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

    s = create_schedule([c.lazydata])[0]
    k = Linearizer(s.ast)
    k.hand_coded_optimizations()  # implicit trigger float4 dim
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 1)

  def test_float4_multidim_unaligned_load(self):
    a = Tensor.rand(2, 9).realize().shrink(((0, 2), (1, 9),))
    b = Tensor.rand(2, 9).realize().shrink(((0, 2), (1, 9),))
    c = a + b

    s = create_schedule([c.lazydata])[0]
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

    s = create_schedule([c.lazydata])[0]
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

    s = create_schedule([c.lazydata])[0]
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

    s = create_schedule([c.lazydata])[0]
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

    s = create_schedule([c.lazydata])[0]
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

    s = create_schedule([c.lazydata])[0]
    k = Linearizer(s.ast)
    k.shift_to(0, 4)  # float4 axis
    k.upcast()
    k.linearize()

    assert TestFloat4.count_float4(k) == (1, 1)

@unittest.skipIf(not isinstance(Device[Device.DEFAULT], Compiled), "linearizer is only for compiled backends")
class TestHandCodedOpts(unittest.TestCase):
  def test_masked_upcast(self):
    layer_1 = Tensor.cat(*[Tensor.rand(5) for _ in range(4)])
    layer_2 = Tensor.cat(layer_1.unsqueeze(0), Tensor.rand(6, 20))

    s = create_schedule([layer_2.lazydata])[-1]
    k = Linearizer(s.ast)
    k.hand_coded_optimizations()
    assert len(k.bufs) == 6  # make sure all ops are done in one kernel
    # masked upcast should upcast masked axis of size 7
    # masked upcast should not upcast large (20) last axis
    # float4/other hcopt shouldn't upcast last axis, since we already have 7 upcast, and the last axis is not very contiguous
    assert k.upcasted == 1 and k.full_shape[-1] == 7

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "Failing because of custom kernel splitting to circumvent the 8 buffer limit")
  def test_masked_upcast_wino(self):
    monster = Tensor.stack([Tensor.stack([Tensor.rand(16) for _ in range(6)]) for _ in range(6)])

    s = create_schedule([monster.lazydata])[-1]
    k = Linearizer(s.ast)
    k.hand_coded_optimizations()
    assert len(k.bufs) == 37  # make sure all ops are done in one kernel
    # should upcast the two Tensor.stacks
    assert k.upcasted >= 2 and k.full_shape[k.shape_len-k.upcasted:k.shape_len].count(6) == 2

  def test_masked_upcast_wino_full(self):
    with Context(WINO=1):
      x,w = Tensor.rand(1,4,8,8, requires_grad=True).realize(), Tensor.rand(4,4,3,3, requires_grad=True).realize()
      out = Tensor.conv2d(x,w, padding=1)
      upcasts = []
      wino_schedule = create_schedule([out.lazydata])
      # collect upcasts of tile transform kernels
      for i, si in enumerate(wino_schedule):
        k = Linearizer(si.ast)
        k.hand_coded_optimizations()
        if k.reduceop is not None: continue  # not a tile transform kernel (there is a gemm reduce kernel)
        if len(k.bufs) < 36: continue  # not a tile transform kernel (there's a permute kernel at the end)
        upcasts.append(tuple(k.full_shape[k.shape_len - k.upcasted:k.shape_len]))
      assert len(upcasts) == 3  # 3 transformation matrices
      assert len(wino_schedule) <= 4  # 4 kernels
      # this test case's inputs are too small, so one of the 4-stacks became a local, which is fine i guess
      assert upcasts.count((6, 6)) == 2 #and upcasts.count((4, 4)) == 1

      out.mean().backward()
      backward_schedule = create_schedule([x.grad.lazydata, w.grad.lazydata])
      for si in backward_schedule:
        k = Linearizer(si.ast)
        k.hand_coded_optimizations()
        k.linearize()
        if len(k.bufs) < 20: continue  # not a tile transform kernel
        # heuristic number to make sure that at least some upcasts but not too many upcasts are being done
        assert 6 <= prod(k.full_shape[k.shape_len - k.upcasted:k.shape_len]) <= 216
      assert len(backward_schedule) <= 13  # just the current number, but it could be better

  def test_masked_upcast_many(self):
    layer_1 = Tensor.cat(Tensor.rand(3, 4), Tensor.rand(4, 4))
    layer_2 = Tensor.cat(layer_1.unsqueeze(0), Tensor.rand(6, 7, 4))
    layer_3 = Tensor.cat(layer_2.unsqueeze(0), Tensor.rand(6, 7, 7, 4))

    s = create_schedule([layer_3.lazydata])[-1]
    k = Linearizer(s.ast)
    k.hand_coded_optimizations()
    assert len(k.bufs) == 5  # make sure all ops are done in one kernel
    # check that we don't do too many upcasts
    assert prod(k.full_shape[k.shape_len-k.upcasted:k.shape_len]) <= 49

  def test_matvec(self):
    if not Device[Device.DEFAULT].compiler.linearizer_opts.has_local:
      self.skipTest("Only devices with locals")
    N = 128
    a = Tensor.rand(1, N).realize()
    b = Tensor.rand(N, N).realize()
    c = a @ b

    s = create_schedule([c.lazydata])[0]
    k = Linearizer(s.ast)
    k.hand_coded_optimizations()

    assert k.group_for_reduces == 1
    assert k.local_dims == 1
    assert k.upcasted == 1

def helper_linearizer_opt(r:Tensor, opts=[], apply_tc=False, atol=1e-4, rtol=1e-4, color_sizes=[]):
  wanna_output = None
  realized_ast, real_bufs = helper_realized_ast(r)

  def check_opt(opts, create_k, to_prg, expected_color_size):
    k = create_k()
    if apply_tc:
      assert k.apply_tensor_cores(1, opts), "no tensor core triggered"
    else:
      for opt in opts:
        k.apply_opt(opt)
    if expected_color_size is not None:
      assert (cs:=[(x,y) for x,y in zip(k.colors(), k.full_shape)]) == expected_color_size, f"expected={expected_color_size} got={cs}"
    prg = to_prg(k)
    real_bufs[0].copyin(np.zeros((real_bufs[0].size, ), dtype=real_bufs[0].dtype.np).data) # Zero to check that all values are filled
    prg.exec(real_bufs)
    np.testing.assert_allclose(wanna_output, np.frombuffer(real_bufs[0].as_buffer(), real_bufs[0].dtype.np), atol=atol, rtol=rtol)

  # Get baseline, which is not optimized at all.
  k = Linearizer(realized_ast)
  prg = Device[Device.DEFAULT].to_program(k)
  prg.exec(real_bufs)
  wanna_output = np.frombuffer(real_bufs[0].as_buffer(), real_bufs[0].dtype.np).copy()

  # Check correctness of handcoded optimiztions.
  k = Linearizer(realized_ast)
  k.hand_coded_optimizations()
  prg = Device[Device.DEFAULT].to_program(k)
  real_bufs[0].copyin(np.zeros((real_bufs[0].size, ), dtype=real_bufs[0].dtype.np).data) # Zero to check that all values are filled
  prg.exec(real_bufs)
  np.testing.assert_allclose(wanna_output, np.frombuffer(real_bufs[0].as_buffer(), real_bufs[0].dtype.np), atol=atol, rtol=rtol)
  for i, x in enumerate(opts): # Check custom transformations if any.
    check_opt(x, lambda: Linearizer(realized_ast), Device[Device.DEFAULT].to_program, color_sizes[i] if i < len(color_sizes) else None)

@unittest.skipIf(not isinstance(Device[Device.DEFAULT], Compiled), "linearizer is only for compiled backends")
class TestLinearizerOpts(unittest.TestCase):
  def test_local_and_grouped_reduce(self):
    if not Device[Device.DEFAULT].compiler.linearizer_opts.has_local or not Device[Device.DEFAULT].compiler.linearizer_opts.has_shared:
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
      # Checking how it works with locals + grouped reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 64)],
      # Checking how it works with locals + grouped reduce + upcasts
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.UPCAST, 0, 8), Opt(OptOps.UNROLL, 1, 4)],
    ])

  def test_upcasts(self):
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
    Tensor.manual_seed(1772)
    a = Tensor.rand(4)
    b = Tensor.rand(4)
    r = (a+b).sqrt() * ((a+1).exp())
    helper_linearizer_opt(r, [
      [Opt(OptOps.UPCAST, 0, 4)], # Checking how it works with upcasts
    ])

  def test_matmul(self):
    if not Device[Device.DEFAULT].compiler.linearizer_opts.has_local or not Device[Device.DEFAULT].compiler.linearizer_opts.has_shared:
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
      # Checking all together
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 4),
       Opt(OptOps.UPCAST, 1, 2)],
      # Full global upcast + local
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 8)],
    ])

  def test_double_reduce(self):
    if not Device[Device.DEFAULT].compiler.linearizer_opts.has_local or not Device[Device.DEFAULT].compiler.linearizer_opts.has_shared:
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
      # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 32), Opt(OptOps.UNROLL, 1, 4)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2),
       Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UNROLL, 1, 4)], # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2),
       Opt(OptOps.UPCAST, 0, 2)], # No globals
    ])

  def test_invalid_tensor_core_extra_opts(self):
    if not Device[Device.DEFAULT].compiler.linearizer_opts.has_tensor_cores:
      self.skipTest("device doesn't have tensor cores")
    if Device.DEFAULT not in tensor_cores:
      self.skipTest("No tensor cores for device")

    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    realized_ast, _ = helper_realized_ast(a@b)
    invalid_opts = [
      [Opt(OptOps.LOCAL, 2, 2)],
      [Opt(OptOps.UPCAST, 2, 2)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 2, 2)],
    ]
    for x in invalid_opts:
      k = Linearizer(realized_ast)
      with self.assertRaises(AssertionError):
        assert k.apply_tensor_cores(use_tensor_cores=1, extra_opts=x), "no valid tensor core" # for METAL in runners

  def test_tensor_core_opts(self):
    if not Device[Device.DEFAULT].compiler.linearizer_opts.has_tensor_cores:
      self.skipTest("device doesn't have tensor cores")
    if Device.DEFAULT not in tensor_cores:
      self.skipTest("No tensor cores for device")

    N = 128
    Tensor.manual_seed(1552)
    for tc in tensor_cores[Device[Device.DEFAULT].compiler.linearizer_opts.device]:
      a, b = Tensor.rand(N, N, dtype=tc.dtype_in), Tensor.rand(N, N, dtype=tc.dtype_in)
      r = a.matmul(b, acc_dtype=tc.dtype_out)
      (atol, rtol) = ((0.25, 0.01) if tc.dtype_out == dtypes.half else (3e-2, 1e-3)) if tc.dtype_in == dtypes.half else (1e-4, 1e-4)
      helper_linearizer_opt(r, [
        [],
        [Opt(OptOps.UPCAST, 0, 4)],
        [Opt(OptOps.UPCAST, 1, 4)],
        [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4)], # check upcasts
        [Opt(OptOps.UNROLL, 0, 2)], # check unroll
        [Opt(OptOps.UNROLL, 0, 0)], # check full unroll of reduce with locals
        [Opt(OptOps.LOCAL, 0, 4)], # check local
        [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 2)], # check combo of unroll and local
        [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 2)],
        [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4)],
        [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.LOCAL, 0, 2)],
        [Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UPCAST, 0, 4)], # check permutations
        [Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 0, 4)],
        [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 1, 4)],
        [Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 4)],
        [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 0, 4)],
        # [Opt(OptOps.GROUP, 0, 2)] # doesn't work because group_for_reduce dims become early locals (conflicting with TC)
      ], apply_tc=True, atol=atol, rtol=rtol)

  def test_padto_matmul(self):
    if Device.DEFAULT in ["CUDA", "HIP"]: self.skipTest("super slow on CUDA and HIP because of the big grid dims")
    N = 17 * 17
    Tensor.manual_seed(289)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    helper_linearizer_opt(a@b, [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 1, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32)],
      # can optimize further post PADTO
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32), Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.UPCAST, 1, 2),],
    ])

  def test_padto_max(self):
    N = 17 * 17
    a = -Tensor.ones(N, N)

    helper_linearizer_opt(a.max(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])
    helper_linearizer_opt(a.max(1), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

    # cannot pad a reduce axis
    with self.assertRaises(AssertionError):
      helper_linearizer_opt(a.max(), [[Opt(OptOps.PADTO, 0, 32)],])
    with self.assertRaises(AssertionError):
      helper_linearizer_opt(a.max(0), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_where(self):
    N = 17 * 17
    a = (Tensor.rand(N, N).max(axis=0, keepdim=True) > 1).where(1, 0)
    helper_linearizer_opt(a.max(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

  def test_color_shapes_with_local(self):
    if not Device[Device.DEFAULT].compiler.linearizer_opts.has_local or not Device[Device.DEFAULT].compiler.linearizer_opts.has_shared:
      self.skipTest("Only Compiled uses linearizer with locals and shared")

    N = 32
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = a@b
    opts_shapes = [
      ([Opt(OptOps.LOCAL, 0, 2)], [("blue",16),("blue",32),("cyan",2),("red",32)]),
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.GROUP, 0, 2)], [("blue",16),("blue",32),("cyan",2),("green",2),("red",16)]),
      # check to ensure local_dims are stable for full UNROLL of first_reduce
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.UNROLL, 0, 0)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.UNROLL, 0, 0),Opt(OptOps.LOCAL, 0, 2)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      # check behavior for full UNROLL on an existing GROUP
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.GROUP, 0, 0),Opt(OptOps.UNROLL, 0, 2)], [("blue",16),("blue",32),("cyan",2),("green",16),("magenta",2)]),
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.GROUP, 0, 0),Opt(OptOps.UNROLL, 0, 0)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.GROUP, 0, 0),Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.UNROLL, 0, 0)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.GROUP, 0, 2),Opt(OptOps.UNROLL, 0, 0)], [("blue",32),("blue",32),("red",16),("magenta",2)]),
    ]
    helper_linearizer_opt(r, [x[0] for x in opts_shapes], color_sizes=[x[1] for x in opts_shapes])

class TestLinearizerHelper(unittest.TestCase):
  def test_num_node_expand(self):
    a = NumNode(42)
    assert expand_node(a) == [a]

  def test_variable_expand(self):
    a = Variable("a", 5, 7)
    assert expand_node(a) == [a]

  def test_variable_expand_expr_none(self):
    a = Variable("_uidx0", 5, 7)
    assert expand_node(a) == [NumNode(5), NumNode(6), NumNode(7)]

  def test_mul_node_expand(self):
    a = Variable("_uidx0", 5, 7)
    m = MulNode(a, 3)
    assert expand_node(m) == [NumNode(15), NumNode(18), NumNode(21)]

    b = Variable("b", 1, 3)
    n = MulNode(b, 3)
    assert expand_node(n) == [Variable("b", 1, 3)*3]

  def test_sum_node_expand(self):
    a = Variable("_uidx0", 1, 3)
    b = Variable("b", 5, 7)
    s1 = a + b
    assert expand_node(s1) == [Node.sum([NumNode(i),b]) for i in range(1,4)]

  def test_multi_expand(self):
    a = Variable("a", 1, 3)
    b = Variable("b", 14, 17)
    s1 = a + b
    # expand increments earlier variables faster than later variables (as specified in the argument)
    # this behavior was just copied from before, no idea why this should be true
    assert expand_node(s1, (a, b)) == [NumNode(x + y) for x in range(b.min, b.max + 1) for y in range(a.min, a.max + 1)]

if __name__ == '__main__':
  unittest.main()
