from typing import List, Tuple
import numpy as np
import unittest
from dataclasses import replace

from tinygrad.codegen.kernel import Opt, OptOps, KernelOptError, tensor_cores
from tinygrad.codegen.linearizer import Linearizer, UOp, UOps, expand_node, expand_idxs
from tinygrad.device import Device, Buffer
from tinygrad.ops import BinaryOps, BufferOps, MemBuffer, ConstBuffer, LazyOp, LoadOps, TernaryOps, ReduceOps, UnaryOps
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import MulNode, Variable, NumNode, Node
from tinygrad.tensor import Tensor
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import run_schedule, lower_schedule, CompiledRunner
from tinygrad.helpers import prod, Context, getenv, CI
from tinygrad.dtype import DType, dtypes
from tinygrad.codegen.uops import UOpGraph

def helper_realized_ast(r:Tensor):
  s = create_schedule([r.lazydata])
  run_schedule(s[:-1])  # run all kernels except the last one
  # now all input LazyBuffers buffers in s[-1] should be realized
  # allocate an output buffer
  output_buffer = Buffer((out:=s[-1].outputs[0]).device, out.size, out.dtype).allocate()
  return s[-1].ast[0], [output_buffer] + list(s[-1].inputs)

def helper_tc_allclose(n:int, m:int, k:int, dtype_in:DType, dtype_out:DType, axis:int=0, tc_opt:int=0):
  a, b = Tensor.rand(m, k, dtype=dtype_in), Tensor.rand(k, n, dtype=dtype_in)
  np_a, np_b = a.numpy(), b.numpy()
  r = a.matmul(b, acc_dtype=dtype_out)
  sched = create_schedule([r.lazydata])
  realized_ast = sched[-1].ast[0]
  run_schedule(sched)
  out = r.numpy()
  k = Linearizer(realized_ast)
  k.apply_tensor_cores(1, axis=axis, tc_opt=tc_opt)
  k.linearize()
  assert len([uop for uop in k.uops if uop.uop is UOps.WMMA]) > 0, "tensor core not triggered"
  assert len([x for x in k.applied_opts if x.op is OptOps.TC]) == 1, "tensor core opt not included"
  np_c = np_a @ np_b
  if dtype_out == dtypes.half: tc_atol, tc_rtol = 1e-2, 1e-3
  elif dtype_in == dtypes.bfloat16: tc_atol, tc_rtol = 1e-2, 3e-3
  else: tc_atol, tc_rtol = 5e-3, 1e-4
  np.testing.assert_allclose(np_c, out, atol=tc_atol, rtol=tc_rtol)

def helper_tc_ensure_uops_and_opts_count(n: int, m:int, k:int, dtype_in:DType, dtype_out:DType, axis:int=0, tc_opt:int=0, ensure_triggered:bool=True):
  a, b = Tensor.rand(m, k, dtype=dtype_in), Tensor.rand(k, n, dtype=dtype_in)
  r = a.matmul(b, acc_dtype=dtype_out)
  sched = create_schedule([r.lazydata])
  realized_ast = sched[-1].ast[0]
  k = Linearizer(realized_ast)
  k.apply_tensor_cores(1, axis=axis, tc_opt=tc_opt)
  k.linearize()
  wmmas = len([uop for uop in k.uops if uop.uop is UOps.WMMA])
  tcs = len([x for x in k.applied_opts if x.op is OptOps.TC])
  if ensure_triggered:
    assert wmmas > 0, "tensor core not triggered"
    assert tcs == 1, "tensor core opt not included"
  else:
    assert wmmas == 0, "tensor core is incorrectly triggered"
    assert tcs == 0, "tensor core opt is incorrectly included"

class TestLinearizer(unittest.TestCase):
  def test_arg_dedup(self):
    a, b = Tensor.randn(4), Tensor.randn(4)
    np_a, np_b = a.numpy(), b.numpy()
    c = ((a.shrink(((0, 2),)) - a.shrink(((2, 4),))) - (b.shrink(((0, 2),)) - b.shrink(((2, 4),))))
    lowered = list(lower_schedule(create_schedule([c.lazydata])))
    for ei in lowered: ei.run()
    rawbufs = lowered[-1].bufs
    assert len(rawbufs) == 3 and set(rawbufs[1:]) == {a.lazydata.base.realized, b.lazydata.base.realized}
    np_c = (np_a[:2] - np_a[2:]) - (np_b[:2] - np_b[2:])
    np.testing.assert_allclose(np_c, c.numpy(), atol=1e-4, rtol=1e-4)

  def test_load_removed(self):
    a = Tensor.rand(1).realize()
    b = Tensor.rand(1).realize()
    ta = Tensor.where(Tensor(True), a, b).numpy()
    tb = Tensor.where(Tensor(False), a, b).numpy()
    np.testing.assert_equal(a.numpy(), ta)
    np.testing.assert_equal(b.numpy(), tb)

  def test_multioutput(self):
    dtype, st = dtypes.int, ShapeTracker.from_shape((8,))
    a = LazyOp(BufferOps.LOAD, arg=MemBuffer(idx=2, dtype=dtype, st=st))
    b = LazyOp(BufferOps.LOAD, arg=MemBuffer(idx=3, dtype=dtype, st=st))
    out0 = LazyOp(BufferOps.STORE, (LazyOp(op=BinaryOps.ADD, src=(a,b)),), MemBuffer(idx=0, dtype=dtype, st=st))
    out1 = LazyOp(BufferOps.STORE, (LazyOp(op=BinaryOps.MUL, src=(a,b)),), MemBuffer(idx=1, dtype=dtype, st=st))

    a_t = Tensor.full(st.shape, 2).contiguous().realize()
    b_t = Tensor.full(st.shape, 3).contiguous().realize()
    lin = helper_linearizer_ast((out0, out1), [a_t, b_t], wanna_output=[a_t.numpy()+b_t.numpy(), a_t.numpy()*b_t.numpy()])[0]

    stores = [u for u in lin.uops if u.uop is UOps.STORE]
    mutable_bufs = [u for u in lin.uops if u.uop is UOps.DEFINE_GLOBAL and u.arg[-1]]
    assert len(mutable_bufs) == len(stores) == 2
    assert [u.arg[0] for u in mutable_bufs] == [0, 1]

  def test_end_local(self):
    if not (opts:=Device[Device.DEFAULT].renderer).has_local or not opts.has_shared: self.skipTest("device does not support locals")
    load = MemBuffer(idx=1, dtype=dtypes.int, st=ShapeTracker.from_shape((32,)))
    store = MemBuffer(idx=0, dtype=dtypes.int, st=ShapeTracker.from_shape((1,)))
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, arg=load),), arg=(0,)),), arg=store),

    load_t = Tensor.full(load.st.shape, 1).contiguous().realize()
    k = helper_linearizer_ast(ast, [load_t], wanna_output=[load_t.numpy().sum()])[1]
    self.assertEqual(k.uops.uops[-1].uop, UOps.ENDIF)
    self.assertLess(k.uops.uops.index([x for x in k.uops.uops if x.uop is UOps.STORE][-1]), k.uops.uops.index(k.uops.uops[-1]))

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_early_end_local(self):
    shape, output_shape = (32,), (1,)
    load0 = MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker.from_shape(shape))
    load1 = MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker.from_shape(shape))
    store = MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker.from_shape(output_shape))
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.ADD, src=(
      LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, arg=load0),), arg=((0,),)),
      LazyOp(op=BufferOps.LOAD, arg=load1), )),), arg=((0,),)),), arg=store)
    k = Linearizer(ast)
    k.hand_coded_optimizations()
    k.linearize()
    self.assertEqual(len(endifs:=[x for x in k.uops.uops if x.uop is UOps.ENDIF]), len(ifs:=[x for x in k.uops.uops if x.uop is UOps.IF]))
    self.assertEqual(len(barriers:=[x for x in k.uops.uops if x.uop is UOps.BARRIER]), 3)
    self.assertEqual(k.uops.uops[k.uops.uops.index(endifs[0])-1].uop, UOps.STORE)
    self.assertEqual(k.uops.uops[k.uops.uops.index(endifs[0])+1], barriers[1])
    self.assertEqual(k.uops.uops[k.uops.uops.index(endifs[0])+2].uop, UOps.LOAD)
    self.assertLess(k.uops.uops.index(barriers[0]), k.uops.uops.index(ifs[0]))
    self.assertLess(k.uops.uops.index(ifs[0]), k.uops.uops.index(endifs[0]))
    self.assertLess(k.uops.uops.index(barriers[1]), k.uops.uops.index(ifs[1]))

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_basic_multireduce(self):
    def gen(shape, axis):
      output_shape = tuple([1 if i == axis else x for i,x in enumerate(list(shape))])
      load0 = MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker.from_shape(shape))
      load1 = MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker.from_shape(shape))
      store = MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker.from_shape(output_shape))
      ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.SUB, src=(
        LazyOp(op=BufferOps.LOAD, src=(), arg=load0),LazyOp(op=ReduceOps.SUM, src=(
        LazyOp(op=BufferOps.LOAD, src=(), arg=load1),), arg=((axis,), dtypes.float))), arg=None),), arg=((axis,), dtypes.float)),), arg=store)
      return [ast]

    for axis in [0, 1, 2]:
      ast = gen((4, 32, 64), axis)
      k = Linearizer(*ast)
      def recursive_reduceops(x: LazyOp): return [c for v in x.src for c in recursive_reduceops(v)] + [v for v in list(x.src) if v.op in ReduceOps]
      for i,r in enumerate(k.reduceops): assert not any([r in recursive_reduceops(x) for x in k.reduceops[:i]]), "reduceops are out of order"
      k.to_program()
      assert (real_outs:=len([u for u in k.uops if u.uop is UOps.STORE])) == 1, f"should have generated 1 BufferOps.STORE but got {real_outs}"
      assert (real_accs:=len([u for u in k.uops if u.uop is UOps.DEFINE_ACC])) == 2, f"should have generated 2 UOps.DEFINE_ACC but got {real_accs}"
      assert (real_loops:=len([u for u in k.uops if u.uop is UOps.LOOP])) == 2, f"should have generated 2 UOps.LOOP but got {real_loops}"

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_multireduce_store_locals(self):
    def gen(shape, axis):
      output_shape = tuple([1 if i == axis else x for i,x in enumerate(list(shape))])
      load0 = MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker.from_shape(shape))
      load1 = MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker.from_shape(shape))
      store = MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker.from_shape(output_shape))
      ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.SUB, src=(
        LazyOp(op=BufferOps.LOAD, src=(), arg=load0),LazyOp(op=ReduceOps.SUM, src=(
        LazyOp(op=BufferOps.LOAD, src=(), arg=load1),), arg=((axis,), dtypes.float))), arg=None),), arg=((axis,), dtypes.float)),), arg=store)
      return [ast]

    for axis in [1, 2]:
      ast = gen((4, 32, 64), axis)
      k = Linearizer(*ast)
      def recursive_reduceops(x: LazyOp): return [c for v in x.src for c in recursive_reduceops(v)] + [v for v in list(x.src) if v.op in ReduceOps]
      for i,r in enumerate(k.reduceops): assert not any([r in recursive_reduceops(x) for x in k.reduceops[:i]]), "reduceops are out of order"
      k.hand_coded_optimizations()
      k.to_program()
      local_buf = [u for u in k.uops if u.uop is UOps.DEFINE_LOCAL][0]
      self.assertEqual(len(real_local_stores:=[u for u in k.uops if u.uop is UOps.STORE and local_buf in u.vin]), 3, \
        f"should have generated 3 BufferOps.STORE to the local buf but got {len(real_local_stores)}")
      self.assertEqual(len(real_local_loads:=[u for u in k.uops if u.uop is UOps.LOAD and local_buf in u.vin]), 3, \
        f"should have generated 3 BufferOps.LOAD to the local buf but got {len(real_local_loads)}")
      self.assertEqual((real_local_stores[1].vin[1].uop, real_local_stores[1].vin[1].arg), (UOps.CONST, 0))
      self.assertEqual((real_local_loads[1].vin[1].uop, real_local_loads[1].vin[1].arg), (UOps.CONST, 0))

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "device does not support upcast")
  def test_multireduce_upcasting(self):
    shape = (2, 7)
    output_shape = (2, 1)
    load0 = MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker.from_shape(shape))
    load1 = MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker.from_shape(shape))
    store = MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker.from_shape(output_shape))
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.SUB, src=(
      LazyOp(op=BufferOps.LOAD, src=(), arg=load0),LazyOp(op=ReduceOps.SUM, src=(
      LazyOp(op=BufferOps.LOAD, src=(), arg=load1),), arg=((1,), dtypes.float))), arg=None),), arg=((1,), dtypes.float)),), arg=store)
    k = Linearizer(ast)
    k.upcast()
    k.linearize()
    k.to_program()
    define_globals = [u for u in k.uops.uops if u.uop is UOps.DEFINE_GLOBAL]
    self.assertEqual(len([u for u in k.uops.uops if u.uop is UOps.DEFINE_ACC]), 2)
    self.assertEqual(len([u for u in k.uops.uops if u.uop is UOps.LOAD and define_globals[1] in u.vin]), 7)
    self.assertEqual(len([u for u in k.uops.uops if u.uop is UOps.LOAD and define_globals[2] in u.vin]), 7)
    self.assertEqual(len([u for u in k.uops.uops if u.uop is UOps.ALU and u.arg is BinaryOps.SUB]), 7)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_multireduce_loop_scope(self):
    shape = (4, 32, 64)
    axis = 2
    output_shape = tuple([1 if i == axis else x for i,x in enumerate(list(shape))])
    load = MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker.from_shape(shape))
    store = MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker.from_shape(output_shape))
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.SUB, src=(
      LazyOp(op=BufferOps.LOAD, src=(), arg=load), LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=load),)
      , arg=((axis,), dtypes.float))), arg=None),), arg=((axis,), dtypes.float)),), arg=store)
    k = Linearizer(ast)
    k.linearize()
    k.to_program()
    def get_recursive_children(x:UOp): return set.union(set(x.vin), *[get_recursive_children(v) for v in x.vin])
    loop = None
    for u in k.uops.uops:
      if u.uop is UOps.LOOP: loop = u
      elif loop is None: continue
      elif u.uop is UOps.ENDLOOP and loop in u.vin: loop = None
      else: self.assertIn(loop, get_recursive_children(u), f"Any uop within a loop should depend on the loop: {u}")

  # @unittest.skip
  def test_mean_std_multireduce(self):
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=UnaryOps.SQRT, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.SUB, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))),), arg=(0, 1, 2)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=7.619047619047618e-05, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None)), arg=None), LazyOp(op=BinaryOps.SUB, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))),), arg=(0, 1, 2)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=7.619047619047618e-05, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None)), arg=None)), arg=None),), arg=(0, 1, 2)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=7.619628162145687e-05, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),))))), arg=None),), arg=None),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),)))), # noqa: E501

    x = Tensor.randn(15, 25, 35).realize()
    helper_linearizer_ast(ast, [x], wanna_output=[x.numpy().std()])

  # @unittest.skip
  def test_mean_std_multireduce(self):
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=UnaryOps.SQRT, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.SUB, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))),), arg=(0, 1, 2)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=7.619047619047618e-05, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None)), arg=None), LazyOp(op=BinaryOps.SUB, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))),), arg=(0, 1, 2)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=7.619047619047618e-05, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None)), arg=None)), arg=None),), arg=(0, 1, 2)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=7.619628162145687e-05, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),))))), arg=None),), arg=None),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),)))), # noqa: E501
    x = Tensor.randn(15, 25, 35).realize()
    helper_linearizer_ast(ast, [x], wanna_output=[x.numpy().std()])

  # @unittest.skip
  def test_mean_std_multireduce_mid_dim(self):
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=UnaryOps.SQRT, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.SUB, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))),), arg=(1,)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.04, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None)), arg=None), LazyOp(op=BinaryOps.SUB, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))),), arg=(1,)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.04, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None)), arg=None)), arg=None),), arg=(1,)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.04, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),))))), arg=None),), arg=None),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 1, 35), strides=(35, 35, 1), offset=0, mask=None, contiguous=True),)))), # noqa: E501
    x = Tensor.randn(15, 25, 35).realize()
    helper_linearizer_ast(ast, [x], wanna_output=[x.numpy().std(1).reshape(-1)])

  # @unittest.skip
  def test_mean_std_multireduce_multiout(self):
    std = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=UnaryOps.SQRT, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.SUB, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))),), arg=(0, 1, 2)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=7.619047619047618e-05, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None)), arg=None), LazyOp(op=BinaryOps.SUB, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))),), arg=(0, 1, 2)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=7.619047619047618e-05, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None)), arg=None)), arg=None),), arg=(0, 1, 2)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=7.619628162145687e-05, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),))))), arg=None),), arg=None),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),)))) # noqa: E501
    mean = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(875, 35, 1), offset=0, mask=None, contiguous=True),)))),), arg=(0, 1, 2)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=7.619047619047618e-05, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(15, 25, 35), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),)))) # noqa: E501
    x = Tensor.randn(15, 25, 35).realize()
    helper_linearizer_ast((std,mean), [x], wanna_output=[x.numpy().std(), x.numpy().mean()])

  def test_load_dedup(self):
    # for different leaves in the AST, the same loads may occur.

    a = Tensor.randn(4).realize()
    # these are of size 3 to avoid float4 coalesce
    r = a[:-1] + a[1:]

    k = Linearizer(*create_schedule([r.lazydata])[-1].ast)
    k.upcast()
    k.linearize()
    num_loads = len([uop for uop in k.uops if uop.uop is UOps.LOAD])
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

    a_bufs = [u.uop for u in lin.uops.uops[-2].vin[0].vin]
    b_bufs = [u.uop for u in lin.uops.uops[-2].vin[1].vin]

    assert a_bufs == [UOps.LOAD, UOps.CONST]
    assert b_bufs == [] # [UOps.CONST, UOps.CONST] will be folded

  def test_upcast_cse(self):
    # when upcasting, within a subtree, there may be common expressions.

    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = a.expand([2]) + b.expand([2])

    k = Linearizer(*create_schedule([r.lazydata])[-1].ast)
    k.upcast()
    k.linearize()
    num_ops = len([uop for uop in k.uops if uop.uop is UOps.ALU])
    assert num_ops <= 1, "more alu uops than needed"

  def test_reduce_upcast(self):
    if not Device[Device.DEFAULT].renderer.supports_float4:
      self.skipTest("device does not support upcast")
    x, w = Tensor.randn((1,1,3)).realize(), Tensor.randn((1,1,2)).realize()
    r = Tensor.conv2d(x,w,padding=1).relu()

    k = Linearizer(*create_schedule([r.lazydata])[-1].ast)
    k.upcast()
    k.upcast()
    k.linearize()
    accs = [u for u in k.uops if u.uop is UOps.DEFINE_ACC]
    stores = [u for u in k.uops if u.uop is UOps.STORE]
    assert len(accs) == 1
    assert len(stores) == 1
    assert stores[0].vin[-1].dtype == accs[0].dtype == dtypes.float.vec(4)

  def test_upcast_with_locals(self):
    if not (opts:=Device[Device.DEFAULT].renderer).has_local or not opts.has_shared or not opts.supports_float4:
      self.skipTest("device does not support upcasted reduce with locals")

    x, y = Tensor.rand(1,128), Tensor.rand(128, 128)
    r = (x@y).relu()
    k = Linearizer(*create_schedule([r.lazydata])[-1].ast)
    k.hand_coded_optimizations()
    k.linearize()

    accs = [u for u in k.uops if u.uop is UOps.DEFINE_ACC]
    stores = [u for u in k.uops if u.uop is UOps.STORE]

    # the first store is to lds and can be upcasted
    assert accs[0].dtype == stores[0].vin[-1].dtype == dtypes.float.vec(4)
    assert stores[0].vin[0].uop is UOps.DEFINE_LOCAL
    # the second store is to gds with no upcasts
    assert accs[1].dtype == stores[1].vin[-1].dtype == dtypes.float
    assert stores[1].vin[0].uop is UOps.DEFINE_GLOBAL

  def test_zero_fold(self):
    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = Tensor.stack([a, b])

    k = Linearizer(*create_schedule([r.lazydata])[-1].ast)
    k.upcast()
    k.linearize()
    num_ops = len([uop for uop in k.uops if uop.uop is UOps.ALU])
    assert num_ops == 0, "more alu uops than needed"

  def test_sum_acc_dtype(self):
    for tensor_dtype, acc_dtype in (
      (dtypes.bool, dtypes.int), (dtypes.int16, dtypes.int), (dtypes.float16, dtypes.float), (dtypes.bfloat16, dtypes.float)):
      a = Tensor([1, 2, 3], dtype=tensor_dtype).sum()
      k = Linearizer(*create_schedule([a.lazydata])[-1].ast)
      k.linearize()
      local = [uop for uop in k.uops if uop.uop is UOps.DEFINE_ACC]
      assert local[0].dtype == acc_dtype

  def test_arg_acc_dtype(self):
    def helper_arg_acc_dtype(c: Tensor, expected_dtype:DType):
      k = Linearizer(*create_schedule([c.lazydata])[-1].ast)
      k.linearize()
      local = [uop for uop in k.uops if uop.uop is UOps.DEFINE_ACC]
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
    if not Device[Device.DEFAULT].renderer.has_tensor_cores:
      self.skipTest("device doesn't have tensor cores")
    for tc in tensor_cores[Device[Device.DEFAULT].renderer.device]:
      if getenv("EMULATE_CUDA") and (tc.dtype_in == dtypes.bfloat16 or tc.dtype_out == dtypes.bfloat16): continue
      helper_tc_allclose(tc.dims[0], tc.dims[1], tc.dims[2], tc.dtype_in, tc.dtype_out, axis=0, tc_opt=0)

  def test_tensor_cores_padded(self):
    if not Device[Device.DEFAULT].renderer.has_tensor_cores:
      self.skipTest("device doesn't have tensor cores")
    for tc in tensor_cores[Device[Device.DEFAULT].renderer.device]:
      if getenv("EMULATE_CUDA") and (tc.dtype_in == dtypes.bfloat16 or tc.dtype_out == dtypes.bfloat16): continue
      pad = 1

      # check that TC is triggered for TC_OPT=2
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]+pad, tc.dims[1]+pad, tc.dims[2]+pad,
                                           tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=True)

      # check that TC is not triggered for TC_OPT<2
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]+pad, tc.dims[1]+pad, tc.dims[2]+pad,
                                           tc.dtype_in, tc.dtype_out, tc_opt=1, ensure_triggered=False)
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]+pad, tc.dims[1]+pad, tc.dims[2]+pad,
                                           tc.dtype_in, tc.dtype_out, tc_opt=0, ensure_triggered=False)

      # check excessive padding doesn't trigger padded TC in TC_OPT=2
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]//4, tc.dims[1], tc.dims[2], tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=False)
      helper_tc_ensure_uops_and_opts_count(tc.dims[0], tc.dims[1]//4, tc.dims[2], tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=False)
      helper_tc_ensure_uops_and_opts_count(tc.dims[0], tc.dims[1], tc.dims[2]//4, tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=False)

      # check correctness
      helper_tc_allclose(tc.dims[0]+pad, tc.dims[1]+pad, tc.dims[2]+pad, tc.dtype_in, tc.dtype_out, tc_opt=2)

  @unittest.skipIf(Device.DEFAULT == "RHIP", "RHIP is really slow here")
  def test_tensor_cores_multi_reduce(self):
    if not Device[Device.DEFAULT].renderer.has_tensor_cores:
      self.skipTest("device doesn't have tensor cores")
    for tc in tensor_cores[Device[Device.DEFAULT].renderer.device]:
      if getenv("EMULATE_CUDA") and (tc.dtype_in == dtypes.bfloat16 or tc.dtype_out == dtypes.bfloat16): continue
      # this will be a M=G16, N=G32, M=G16, M=G16, K=R16, K=R16, K=R16 with 9 choices of TC MNK axes
      golden_result = None
      for axis in range(9):
        a = Tensor.rand(16, 16, 29, 29, dtype=tc.dtype_in).realize()
        b = Tensor.rand(32, 16, 16, 16, dtype=tc.dtype_in).realize()
        c = a.conv2d(b, padding=1, acc_dtype=tc.dtype_out)
        realized_ast, real_bufs = helper_realized_ast(c)

        k = Linearizer(realized_ast)
        k.apply_tensor_cores(1, axis=axis, tc_opt=2)
        k.linearize()
        assert len([uop for uop in k.uops if uop.uop is UOps.WMMA]) > 0, "tensor core not triggered"
        assert len([x for x in k.applied_opts if x.op is OptOps.TC]) == 1, "tensor core opt not included"

        prg = CompiledRunner(k.to_program())
        real_bufs[0].copyin(np.zeros((real_bufs[0].size, ), dtype=real_bufs[0].dtype.np).data) # Zero to check that all values are filled
        prg.exec(real_bufs)
        result = np.frombuffer(real_bufs[0].as_buffer(), real_bufs[0].dtype.np)

        # ensure the results for each choice of axis matches
        if golden_result is None: golden_result = np.frombuffer(real_bufs[0].as_buffer(), real_bufs[0].dtype.np)
        np.testing.assert_allclose(result, golden_result, atol=0.1, rtol=0.15)

      # check that get_linearizer_actions produces all 9 options
      from tinygrad.features.search import get_linearizer_actions
      tc_actions = [k for i, k in get_linearizer_actions(Linearizer(realized_ast), False).items() if k.applied_opts[0].op == OptOps.TC]
      assert len(tc_actions) == 9, f"get_linearizer_actions should contain 9 possible TC actions, only got {len(tc_actions)}"

  def test_limit_dims_to_max_5d_global(self):
    t = Tensor.empty(3, 4, 5, 6, 7).pad(((1, 1), (1, 1), (1, 1), (1, 1), (1, 1))) + 1
    sched = [si for si in create_schedule([t.lazydata]) if si.ast[0].op not in LoadOps]
    assert len(sched) == 1
    lin = Linearizer(*sched[0].ast)
    assert lin.full_shape[:lin.global_dims] == (5, 6, 7, 8, 9)
    lin.limit_dims_to_max(global_max=[16, 16, 16], local_max=[16, 16, 16])

  def test_sum_collapse(self):
    t = Tensor([2]).reshape(1, 1).expand(256, 256).sum()
    sched = [si for si in create_schedule([t.lazydata]) if si.ast[0].op not in LoadOps]
    assert len(sched) == 1
    lin = Linearizer(*sched[0].ast)
    assert not any(u.uop is UOps.LOOP for u in lin.linearize().uops), "found loop in sum collapse"

  def test_assign_fold(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    m = Tensor.ones(4, 4).shrink(((1, 2), None)).pad(((1, 2), None))
    a.assign(a+m)
    a.realize()
    np.testing.assert_equal(a.flatten().numpy(), [1.,1.,1.,1.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.,1.,1.])

  def test_where_fold(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.shrink(((1, 2), None)).pad(((1, 2), None))
    a.assign(b.where(2, a))
    sched = create_schedule([a.lazydata])
    assert len(sched) == 1
    sched_copy = sched[:]
    run_schedule(sched)
    np.testing.assert_equal(a.flatten().numpy(), [1.,1.,1.,1.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.,1.,1.])
    lin = Linearizer(*sched_copy[-1].ast)
    lin.hand_coded_optimizations()
    lin.linearize()
    assert not any(u.arg == TernaryOps.WHERE for u in lin.uops), "found where where where should be folded"

  def test_simplify_uop(self):
    def helper_test_simplify(uop, dtype, vin, arg=None):
      ast = LazyOp(BufferOps.CONST, (),
                   ConstBuffer(42, dtypes.float, ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),))))
      ast = LazyOp(BufferOps.STORE, (ast,),
                   MemBuffer(0, dtypes.float, ShapeTracker(views=(View(shape=(), strides=(), offset=0, mask=None, contiguous=True),))))
      lin = Linearizer(ast) # this is a dummy ast

      lin.uops = UOpGraph()
      return lin.uops.add(uop, dtype, vin, arg)

    c0 = UOp(UOps.CONST, dtypes.float, vin=(), arg=0.0)
    assert helper_test_simplify(UOps.ALU, dtypes.float, vin=(UOp(UOps.CONST, dtypes.bool, vin=(), arg=True), c0, c0), arg=TernaryOps.WHERE) == c0

    c0 = UOp(UOps.CONST, dtypes.float, vin=(), arg=0.0)
    c1 = UOp(UOps.CONST, dtypes.float, vin=(), arg=1.0)
    assert helper_test_simplify(UOps.ALU, dtypes.float, vin=(UOp(UOps.CONST, dtypes.bool, vin=(), arg=True), c0, c1),
                                arg=TernaryOps.WHERE).arg == c0.arg

  def test_phi_simplification(self):
    def helper(t, max_ops=0):
      sched = create_schedule([t.lazydata])
      assert len(sched) == 1
      k = Linearizer(*sched[0].ast)
      k.hand_coded_optimizations()
      uops = list(k.linearize().uops)
      # ignore kernel optimized IF/LOOP statements for now
      if if_op:=next((u for u in uops if u.uop is UOps.IF), None):
        uops = uops[:uops.index(if_op)]
      assert len(set([u.uop for u in uops if u.uop in {UOps.LOOP, UOps.SPECIAL}])) == 1, "has either specials or loops, not both"
      assert len([u for u in uops if u.uop is UOps.PHI]) == 0, "PHI should have been simplified"
      assert len([u for u in uops if u.arg is BinaryOps.MAX]) <= max_ops, "no unnecessary MAX ops"

    helper(Tensor.arange(5.5, (3.5*300), 3.5))
    helper(Tensor.arange(-1, -100, -5))
    helper(Tensor.arange(-3.2, 6.7, 0.64))
    helper(Tensor.arange(256), max_ops=2)
    helper(Tensor.arange(255), max_ops=0)

@unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "need backends that support float4")
class TestFloat4(unittest.TestCase):
  @staticmethod
  def count_float4(k):
    return (len([uop for uop in k.uops if uop.uop is UOps.LOAD and uop.dtype == dtypes.float.vec(4)]),
            len([uop for uop in k.uops if uop.uop is UOps.STORE and len(uop.vin) == 3 and uop.vin[2].dtype == dtypes.float.vec(4)]))

  # TODO: express opts below as auto opts

  def test_float4_basic(self):
    a = Tensor.rand(2, 8).realize()
    b = Tensor.rand(2, 8).realize()
    c = a + b

    s = create_schedule([c.lazydata])[0]
    k = Linearizer(*s.ast)
    k.hand_coded_optimizations()
    k.linearize()

    assert TestFloat4.count_float4(k) == (2, 1)

  def test_float4_multidim(self):
    a = Tensor.rand(2, 8).realize()
    b = Tensor.rand(2, 8).realize()
    c = a + b

    s = create_schedule([c.lazydata])[0]
    k = Linearizer(*s.ast)
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
    k = Linearizer(*s.ast)
    k.hand_coded_optimizations()  # implicit trigger float4 dim
    k.linearize()

    assert TestFloat4.count_float4(k) == (0, 1)

  def test_float4_multidim_unaligned_load(self):
    a = Tensor.rand(2, 9).realize().shrink(((0, 2), (1, 9),))
    b = Tensor.rand(2, 9).realize().shrink(((0, 2), (1, 9),))
    c = a + b

    s = create_schedule([c.lazydata])[0]
    k = Linearizer(*s.ast)
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
    k = Linearizer(*s.ast)
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
    k = Linearizer(*s.ast)
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
    k = Linearizer(*s.ast)
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
    k = Linearizer(*s.ast)
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
    k = Linearizer(*s.ast)
    k.shift_to(0, 4)  # float4 axis
    k.upcast()
    k.linearize()

    assert TestFloat4.count_float4(k) == (1, 1)

class TestHandCodedOpts(unittest.TestCase):
  def test_masked_upcast(self):
    layer_1 = Tensor.cat(*[Tensor.rand(5) for _ in range(4)])
    layer_2 = Tensor.cat(layer_1.unsqueeze(0), Tensor.rand(6, 20))

    s = create_schedule([layer_2.lazydata])[-1]
    k = Linearizer(*s.ast)
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
    k = Linearizer(*s.ast)
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
        k = Linearizer(*si.ast)
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
        k = Linearizer(*si.ast)
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
    k = Linearizer(*s.ast)
    k.hand_coded_optimizations()
    assert len(k.bufs) == 5  # make sure all ops are done in one kernel
    # check that we don't do too many upcasts
    assert prod(k.full_shape[k.shape_len-k.upcasted:k.shape_len]) <= 49

  def test_matvec(self):
    if not Device[Device.DEFAULT].renderer.has_local:
      self.skipTest("Only devices with locals")
    N = 128
    a = Tensor.rand(1, N).realize()
    b = Tensor.rand(N, N).realize()
    c = a @ b

    s = create_schedule([c.lazydata])[0]
    k = Linearizer(*s.ast)
    k.hand_coded_optimizations()

    assert k.group_for_reduces == 1
    assert k.local_dims == 1
    assert k.upcasted == 1

def helper_linearizer_ast(ast:Tuple[LazyOp, ...], inputs:List[Tensor], *args, **kwargs):
  inbufs = [x.lazydata.buffer for x in inputs]
  outbufs = [Buffer(inbufs[-1].device, out.arg.st.size, out.arg.dtype).allocate() for out in ast]
  return _helper_linearizer_opt_ast(ast, outbufs+inbufs, *args, **kwargs)

def helper_linearizer_opt(r:Tensor, *args, **kwargs):
  realized_ast, real_bufs = helper_realized_ast(r)
  return _helper_linearizer_opt_ast((realized_ast, ), real_bufs, *args, **kwargs)

def _helper_linearizer_opt_ast(realized_ast:Tuple[LazyOp, ...], real_bufs:List[Buffer], opts=[],
                               apply_tc=False, atol=1e-4, rtol=1e-4, color_sizes=[], wanna_output=[]) -> List[Linearizer]:
  lins: List[Linearizer] = []
  outbufs = [real_bufs[i] for i in range(len(realized_ast))]

  def get_prg(k:Linearizer): return CompiledRunner(replace(k.to_program(), dname=Device.DEFAULT))

  def check_opt(opts, create_k, expected_color_size):
    k = create_k()
    lins.append(k)
    if apply_tc:
      assert k.apply_tensor_cores(1, extra_opts=opts), "no tensor core triggered"
    else:
      for opt in opts:
        k.apply_opt(opt)
    if expected_color_size is not None:
      assert (cs:=[(x,y) for x,y in zip(k.colors(), k.full_shape)]) == expected_color_size, f"expected={expected_color_size} got={cs}"
    prg = get_prg(k)
    for buf in outbufs: buf.copyin(np.zeros((buf.size, ), dtype=buf.dtype.np).data) # Zero to check that all values are filled
    prg.exec(real_bufs)

    for i, buf in enumerate(outbufs):
      np.testing.assert_allclose(np.frombuffer(buf.as_buffer(), buf.dtype.np), wanna_output[i], atol=atol, rtol=rtol)

  # Get baseline if it is not provided, which is not optimized at all.
  k = Linearizer(*realized_ast)
  lins.append(k)
  prg = get_prg(k)
  prg.exec(real_bufs)
  if len(wanna_output) == 0: wanna_output = [np.frombuffer(buf.as_buffer(), buf.dtype.np).copy() for buf in outbufs]
  else:
    for i, buf in enumerate(outbufs):
      np.testing.assert_allclose(np.frombuffer(buf.as_buffer(), buf.dtype.np), wanna_output[i], atol=atol, rtol=rtol)

  # Check correctness of handcoded optimiztions.
  k = Linearizer(*realized_ast)
  lins.append(k)
  k.hand_coded_optimizations()
  prg = get_prg(k)
  for buf in outbufs: buf.copyin(np.zeros((buf.size, ), dtype=buf.dtype.np).data) # Zero to check that all values are filled
  prg.exec(real_bufs)
  for i, buf in enumerate(outbufs):
    np.testing.assert_allclose(np.frombuffer(buf.as_buffer(), buf.dtype.np), wanna_output[i], atol=atol, rtol=rtol)
  for i, x in enumerate(opts): # Check custom transformations if any.
    check_opt(x, lambda: Linearizer(*realized_ast), color_sizes[i] if i < len(color_sizes) else None)
  return lins

class TestKernelOpts(unittest.TestCase):
  def test_local_and_grouped_reduce(self):
    if not Device[Device.DEFAULT].renderer.has_local or not Device[Device.DEFAULT].renderer.has_shared:
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
    if not Device[Device.DEFAULT].renderer.has_local or not Device[Device.DEFAULT].renderer.has_shared:
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
    if not Device[Device.DEFAULT].renderer.has_local or not Device[Device.DEFAULT].renderer.has_shared:
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
    if not Device[Device.DEFAULT].renderer.has_tensor_cores:
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

  def test_buf_index_not_found_tensor_core(self):
    if not Device[Device.DEFAULT].renderer.has_tensor_cores:
      self.skipTest("device doesn't have tensor cores")
    if Device.DEFAULT not in tensor_cores:
      self.skipTest("No tensor cores for device")

    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=BinaryOps.CMPEQ, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.int, st=ShapeTracker(views=(View(shape=(1243, 256), strides=(0, 1), offset=0, mask=None, contiguous=False),)))), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.int, st=ShapeTracker(views=(View(shape=(1243, 256), strides=(1, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=(dtypes.float, False)), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=3, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1243, 256), strides=(1, 0), offset=0, mask=None, contiguous=False),))))), arg=None),), arg=(0,)),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 256), strides=(0, 1), offset=0, mask=None, contiguous=True),))))  # noqa: E501
    k = Linearizer(ast, opts=Device[Device.DEFAULT].renderer)
    with self.assertRaises(KernelOptError):
      k.apply_opt(Opt(OptOps.TC, 0, 1))

  def test_tensor_core_opts(self):
    if not Device[Device.DEFAULT].renderer.has_tensor_cores:
      self.skipTest("device doesn't have tensor cores")
    if Device.DEFAULT not in tensor_cores:
      self.skipTest("No tensor cores for device")

    N = 128
    Tensor.manual_seed(1552)
    for tc in tensor_cores[Device[Device.DEFAULT].renderer.device]:
      # bf16 buffer returns float32 numpy outputs so test would fail. testing opt with half suffices.
      if tc.dtype_in == dtypes.bfloat16: continue
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
    if CI and Device.DEFAULT in ["CUDA", "RHIP"]: self.skipTest("super slow on CUDA and RHIP because of the big grid dims")
    N = 17 * 17
    Tensor.manual_seed(289)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    helper_linearizer_opt(a@b, [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 1, 32)],
      [Opt(OptOps.PADTO, 2, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32), Opt(OptOps.PADTO, 2, 32)],
      # can optimize further post PADTO
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32), Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.UPCAST, 1, 2),],
    ])

  def test_padto_upcasted_not_ok(self):
    N = 4
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    helper_linearizer_opt(a@b, [
      [Opt(OptOps.UPCAST, 0, 0)],
      [Opt(OptOps.UPCAST, 1, 0)],
      [Opt(OptOps.UNROLL, 0, 0)],
      [Opt(OptOps.PADTO, 0, 8)],
      [Opt(OptOps.PADTO, 1, 8)],
      [Opt(OptOps.PADTO, 2, 8)],
    ])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.UPCAST, 0, 0), Opt(OptOps.PADTO, 2, 8)]])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.UPCAST, 1, 0), Opt(OptOps.PADTO, 2, 8)]])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.UNROLL, 0, 0), Opt(OptOps.PADTO, 2, 8)]])

  def test_padto_sum_ok(self):
    N = 18 * 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one dimension
    a = Tensor.rand(N, N).shrink(((0, 17), (0, 17))) * 100

    helper_linearizer_opt(a.sum(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])
    helper_linearizer_opt(a.sum(1), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

    # can pad sum reduce axis if there's no unsafe ops prior to sum
    helper_linearizer_opt(a.sum(), [[Opt(OptOps.PADTO, 0, 32)],])
    helper_linearizer_opt(a.sum(0), [[Opt(OptOps.PADTO, 1, 32)],])

    # having unsafe ops after sum is fine
    helper_linearizer_opt(a.sum().exp(), [[Opt(OptOps.PADTO, 0, 32)],])
    helper_linearizer_opt(a.sum(0).exp(), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_sum_not_ok(self):
    N = 18 * 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one dimension
    a = Tensor.rand(N, N).shrink(((0, 17), (0, 17))).exp()
    # exp is not safe to pad
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a.exp().sum(), [[Opt(OptOps.PADTO, 0, 32)],])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a.exp().sum(0), [[Opt(OptOps.PADTO, 1, 32)],])

    b = a < -1
    # lt is not safe to pad
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(b.sum(), [[Opt(OptOps.PADTO, 0, 32)],])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(b.sum(0), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_max(self):
    N = 18 * 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one axis
    a = -Tensor.rand(N, N).shrink(((0, 17), (0, 17))) * 100

    helper_linearizer_opt(a.max(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])
    helper_linearizer_opt(a.max(1), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

    # cannot pad max kernel on reduce
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a.max(), [[Opt(OptOps.PADTO, 0, 32)],])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a.max(0), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_where(self):
    N = 17 * 17
    a = (Tensor.empty(N, N).max(axis=0, keepdim=True) > 1).where(1, 0)
    helper_linearizer_opt(a.max(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

  def test_color_shapes_with_local(self):
    if not Device[Device.DEFAULT].renderer.has_local or not Device[Device.DEFAULT].renderer.has_shared:
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

  def test_expand_nonpresent_var(self):
    a = Variable("a", 1, 3)
    n = NumNode(3) * Variable("b", 1, 3)
    assert expand_node(n, (a,)) == [n, n, n]

  def test_expand_idxs(self):
    uidx0 = Variable("_uidx0", 0, 6)
    uidx1 = Variable("_uidx1", 0, 1)
    idxs = (uidx0 // 5, uidx0 * 5, uidx1)
    assert expand_idxs(idxs) == (uidx0, NumNode(0), uidx1)

class TestLinearizerUOptimize(unittest.TestCase):
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "device doesn't support float4")
  def test_grouped_store_phis(self):
    x, y = Tensor.randn(64,64), Tensor.randn(64,64)
    out = x.matmul(y)

    k = Linearizer(*create_schedule([out.lazydata])[-1].ast)
    k.hand_coded_optimizations()
    k.linearize()

    # check that the float4 cast collapses
    store_vals = [u.vin[-1] for u in k.uops if u.uop is UOps.STORE]
    for val in store_vals:
      assert val.dtype == dtypes.float.vec(4) and val.uop is not UOps.CAST

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "device doesn't support float4")
  def test_grouped_store_values(self):
    x = Tensor.randn((4,3,6,6)).realize()
    out = x.flip((0,1)).contiguous()

    k = Linearizer(*create_schedule([out.lazydata])[-1].ast)
    k.hand_coded_optimizations()
    k.linearize()

    store_val = [u.vin[-1] for u in k.uops if u.uop is UOps.STORE][0]
    assert store_val.dtype == dtypes.float.vec(4) and store_val.uop is not UOps.CAST

  def test_grouped_store_locals_and_globals(self):
    if not Device[Device.DEFAULT].renderer.has_local or not Device[Device.DEFAULT].renderer.has_shared or \
       not Device[Device.DEFAULT].renderer.supports_float4:
      self.skipTest("Only Compiled uses linearizer with locals, shared, and float4")

    x, y = Tensor.rand(128, 128), Tensor.rand(128, 128)
    out = x@y

    opts = [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8),
            Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 2)] # upcast accs in both reduces
    k = Linearizer(*create_schedule([out.lazydata])[-1].ast)
    for opt in opts: k.apply_opt(opt)
    k.linearize()

    local_stores = [u for u in k.uops if u.uop is UOps.STORE and u.vin[0].uop is UOps.DEFINE_LOCAL]
    barrier = [u for u in k.uops if u.uop is UOps.BARRIER][0]
    global_stores = [u for u in k.uops if u.uop is UOps.STORE and u.vin[0].uop is UOps.DEFINE_GLOBAL]

    # check that the float4 cast collapses for all stores
    for store in local_stores+global_stores:
      assert store.vin[-1].dtype == dtypes.float.vec(2) and store.vin[-1].uop is not UOps.CAST
    # check the children's vins
    assert barrier.vin == tuple(local_stores)
    assert len([u for u in k.uops if u.uop is UOps.IF and u.vin[-1] == barrier]) == 1

  def test_grouped_store_local_only(self):
    if not Device[Device.DEFAULT].renderer.has_local or not Device[Device.DEFAULT].renderer.has_shared or \
       not Device[Device.DEFAULT].renderer.supports_float4:
      self.skipTest("Only Compiled uses linearizer with locals, shared, and float4")

    x, y = Tensor.rand(1,128), Tensor.rand(128, 128)
    r = (x@y).relu()
    k = Linearizer(*create_schedule([r.lazydata])[-1].ast)
    k.hand_coded_optimizations()
    k.linearize()

    stores = [u for u in k.uops if u.uop is UOps.STORE]

    # the float4 value stores directly in lds and we skip upcast
    assert stores[0].vin[-1].dtype == dtypes.float.vec(4)
    assert stores[0].vin[-1].uop is not UOps.CAST

    # the global store doesn't change
    assert stores[1].vin[-1].dtype == dtypes.float

  def test_skip_unmatching_upcasts(self):
    if not Device[Device.DEFAULT].renderer.has_local or not Device[Device.DEFAULT].renderer.supports_float4:
      self.skipTest("Needs locals and float4")
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(240, 40, 1, 1), strides=(1, 240, 0, 0), offset=0, mask=None, contiguous=False),)))),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(240, 40, 1, 1), strides=(40, 1, 0, 0), offset=0, mask=None, contiguous=True),)))) # noqa: E501
    opts = [
        Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=0, amt=16),
        Opt(op=OptOps.LOCAL, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=2)
    ]

    k = Linearizer(ast)
    for opt in opts: k.apply_opt(opt)
    k.linearize()

    out = [u for u in k.uops if u.uop is UOps.STORE][0]
    assert out.vin[-1].uop is UOps.CAST and out.vin[-1].dtype == dtypes.float.vec(4)

  def test_skip_unmatching_upcasts_with_gep(self):
    if not Device[Device.DEFAULT].renderer.has_local or not Device[Device.DEFAULT].renderer.supports_float4:
      self.skipTest("Needs locals and float4")
    ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(8, 32, 1, 1), strides=(1, 8, 0, 0), offset=0, mask=None, contiguous=False),)))),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(8, 32, 1, 1), strides=(32, 1, 0, 0), offset=0, mask=None, contiguous=True),)))) # noqa: E501
    opts = [Opt(op=OptOps.LOCAL, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=2, amt=2), Opt(op=OptOps.LOCAL, axis=1, amt=8),
            Opt(op=OptOps.UPCAST, axis=1, amt=0), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=0, amt=8),
            Opt(op=OptOps.UPCAST, axis=1, amt=0), Opt(op=OptOps.UPCAST, axis=0, amt=2)]

    k = Linearizer(ast)
    for opt in opts: k.apply_opt(opt)
    k.linearize()

    out = [u for u in k.uops if u.uop is UOps.STORE][0]
    assert out.vin[-1].uop is UOps.CAST and out.vin[-1].dtype == dtypes.float.vec(2)


if __name__ == '__main__':
  unittest.main()
