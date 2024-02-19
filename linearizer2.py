import numpy as np
from dataclasses import dataclass
import graphlib, unittest, copy
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from tinygrad.codegen.kernel import LocalBuffer
from tinygrad.codegen.uops import UOp, UOps
from tinygrad.device import Buffer, BufferCopy, Compiled, Compiler, Device, JITRunner
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.features.graph import print_tree
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import BinaryOps, BufferOps, ConstBuffer, LazyOp, LoadOps, MemBuffer, Op, ReduceOps
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor
from verify import verify, f32_to_bits
from tinygrad.helpers import panic
from tinygrad.codegen.linearizer import MiniLinearizer as LinearizerOld

def create_graph(outs: List[LazyOp]):
  ts = graphlib.TopologicalSorter()
  def _recursive_add(op: LazyOp):
    for src in op.src: _recursive_add(src)
    ts.add(op, *op.src)
  for out in outs: _recursive_add(out)
  return tuple(ts.static_order())

class ASTRunner(JITRunner):
  def __init__(self, ast: Tuple[LazyOp,...]):
    self.ast = ast
    self.device, self.compiler = cast(Compiled, Device[Device.DEFAULT]), cast(Compiler, Device[Device.DEFAULT].compiler)
    super().__init__()
  def __call__(self, rawbufs: List[Buffer], var_vals, wait=False, jit=False):
    lin = MiniLinearizer(self.ast)
    lin.linearize()
    lib = self.compiler.compile(self.compiler.render("test", lin.uops))
    prg = self.device.runtime("test", lib)
    prg(*[x._buf for x in rawbufs])

@dataclass(frozen=True)
class MiniScheduleItem:
  runner: JITRunner
  rawbufs: List[Buffer]

class Scheduler:
  def __init__(self) -> None:
    self.sched: List[MiniScheduleItem] = []
    self.rawbufs: List[Buffer] = []
    self.host_bufs: List[Buffer] = []

  def create_schedule(self, lb:LazyBuffer):
    if lb.base.op == LoadOps.COPY:
      host_buf, device_buf = cast(Buffer,lb.base.srcs[0].realized), Buffer(lb.device, lb.size, lb.dtype)
      self.sched.append(MiniScheduleItem(BufferCopy(), [device_buf, host_buf]))
      self.rawbufs.append(device_buf)
      self.host_bufs.append(host_buf)
    else:
      ast = graphlib.TopologicalSorter()
      for src in lb.srcs: self.create_schedule(src)
      src = self._recursive_lazyop(lb, ast)
      assert src is not None
      op = LazyOp(BufferOps.STORE, (src, ), MemBuffer(len(self.rawbufs), lb.dtype, lb.st.simplify().unbind()[0]))
      store_buf = Buffer(lb.device, lb.size, lb.dtype)
      lb.realized = store_buf
      self.rawbufs.append(store_buf)
      ast.add(op, src)
      output_ast = tuple(ast.static_order())
      self.sched.append(MiniScheduleItem(ASTRunner(output_ast), self.rawbufs))

  def _recursive_lazyop(self, lb:LazyBuffer, ast) -> LazyOp:
    # these ops have special sources
    if lb.base.op == LoadOps.EMPTY: # LoadOps.EMPTY "defines" the input MemBuffer in the AST
      assert isinstance(lb.realized, Buffer)
      unbound_st, st_var_vals = lb.st.simplify().unbind()
      assert st_var_vals == {}, "variables not supported yet"
      return LazyOp(BufferOps.LOAD, (), MemBuffer(self.host_bufs.index(lb.realized), lb.dtype, unbound_st))
    if lb.base.op == LoadOps.CONST: # Consts are always generated
      return LazyOp(BufferOps.CONST, src=(), arg=ConstBuffer(val=lb.base.arg, dtype=lb.base.dtype, st=lb.st.simplify().unbind()[0]))
    elif lb.base.op == LoadOps.COPY: return self._recursive_lazyop(lb.srcs[0], ast)

    srcs: List[LazyOp] = []
    for src in lb.base.srcs:
      src_op = self._recursive_lazyop(src, ast)
      if src_op is not None:
        srcs.append(src_op)
    op = LazyOp(cast(Op,lb.base.op), src=tuple(srcs))
    ast.add(op, *srcs)
    return op

class MiniLinearizer:
  def __init__(self, ast):
    self.ast = ast
    self.uops: List[UOp] = []
    
    self.buf_pointers: Dict[Union[MemBuffer,LocalBuffer], UOp] = {}

    self.loaded_bufs: Dict[Union[MemBuffer,LocalBuffer], UOp] = {}
    self.alu_cache: Dict[Any, UOp] = {}
    self.reduce_cache: Dict[LazyOp, UOp] = {}

  def const(self, val, dtype=dtypes.int):
    existing = [u for u in self.uops if u.uop == UOps.CONST and u.arg == val]
    if len(existing) != 0: return existing[0]
    uop = UOp(UOps.CONST, dtype=dtype, arg=val)
    self.uops.append(uop)
    return uop

  def _lower_op(self, op:LazyOp) -> UOp:
    if op.op == BufferOps.LOAD: return self.loaded_bufs[op.arg]
    if op.op == BufferOps.CONST: return self.const(op.arg.val, op.arg.dtype)
    if op.op in ReduceOps:
      if op in self.reduce_cache: return self.reduce_cache[op]
      buf: MemBuffer = op.src[0].arg
      reduce_dims = [Variable(f"ridx{i}", 0, dim) for i, dim in enumerate(buf.st.shape)]
      idx = UOp(UOps.LOOP, dtype=dtypes.int, vin=(self.const(reduce_dims[0].min),self.const(reduce_dims[0].max)))
      loop_uops = [idx]
      for i, dim in enumerate(reduce_dims[1:]):
        outer_alu = UOp(UOps.ALU, dtype=dtypes.int, vin=(idx,self.const(reduce_dims[i-1].max)), arg=BinaryOps.MUL)
        inner_loop = UOp(UOps.LOOP, dtype=dtypes.int, vin=(self.const(dim.min),self.const(dim.max)))
        idx = UOp(UOps.ALU, dtype=dtypes.int, vin=(outer_alu,inner_loop), arg=BinaryOps.ADD)
        loop_uops += [inner_loop, outer_alu, idx]
      src = UOp(UOps.LOAD, dtype=buf.dtype, vin=(self.buf_pointers[buf],idx))
      acc = UOp(UOps.DEFINE_ACC, dtype=src.dtype, arg=0)
      reduce_alu = UOp(UOps.ALU, dtype=src.dtype, vin=(acc,src), arg=BinaryOps.ADD if op.op == ReduceOps.SUM else BinaryOps.MAX)
      ret = UOp(UOps.PHI, dtype=src.dtype, vin=(acc,reduce_alu,*loop_uops))
      loop_uops = [acc, *loop_uops, src, reduce_alu, ret, *[UOp(UOps.ENDLOOP, vin=(uop,)) for uop in loop_uops if uop.uop == UOps.LOOP]]
      self.uops.extend(loop_uops)
      self.reduce_cache[op] = ret
      return ret
    srcs = tuple(self._lower_op(src) for src in op.src)
    ret = UOp(UOps.ALU, vin=srcs, dtype=srcs[-1].dtype, arg=op.op)
    key = (ret.vin, ret.arg)
    if key in self.alu_cache: return self.alu_cache[key]
    self.uops.append(ret)
    self.alu_cache[key] = ret
    return ret

  def linearize(self) -> List[UOp]:
    for op in self.ast:
      if not (op.op in BufferOps and isinstance(buf:=op.arg, MemBuffer)): continue
      if buf not in self.buf_pointers:
        self.buf_pointers[buf] = UOp(UOps.DEFINE_GLOBAL, dtype=PtrDType(buf.dtype), arg=f"data{buf.idx}")
        self.uops.append(self.buf_pointers[buf])
      if op.op == BufferOps.LOAD and buf not in self.loaded_bufs:
        self.loaded_bufs[buf] = UOp(UOps.LOAD, dtype=buf.dtype, vin=(self.buf_pointers[buf],self.const(0)))
        self.uops.append(self.loaded_bufs[buf])
      else:
        ret = self._lower_op(op.src[0])
        self.uops.append(UOp(UOps.STORE, dtype=ret.dtype, vin=(self.buf_pointers[buf],self.const(0),ret)))
    return self.uops

class TestLinearizer2(unittest.TestCase):
  def _new_realize(self, x):
    scheduler = Scheduler()
    scheduler.create_schedule(x.lazydata)
    for si in scheduler.sched:
      si.runner(si.rawbufs, var_vals={})
    ret = x.numpy()
    x.lazydata.realized = None
    return ret

  def test_add_simple(self):
    x = Tensor([1]) + Tensor([67])
    ret = self._new_realize(x)
    expected = x.numpy()
    np.testing.assert_equal(ret, expected)

  def test_multi_output_simple(self):
    a = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=0, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    b = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    c = LazyOp(BinaryOps.ADD, src=(a,b))
    out0 = LazyOp(BufferOps.STORE, src=(LazyOp(BinaryOps.ADD, src=(a,c)),), arg=MemBuffer(idx=2, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    out1 = LazyOp(BufferOps.STORE, src=(LazyOp(BinaryOps.MUL, src=(a,b)),), arg=MemBuffer(idx=3, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    graph = create_graph([out0, out1])
    uops = MiniLinearizer(graph).linearize()
    alloc_data, init_outputs, prg, get_outputs = verify(uops)
    a, b = alloc_data([1]), alloc_data([2])
    outs = init_outputs([1,1])
    prg(a, b, *outs, global_size=(1,1,1), local_size=(1,1,1))
    assert get_outputs(outs) == [[4], [2]]

  def test_multi_output_multi_reduce(self):
    a = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=0, dtype=dtypes.int, st=ShapeTracker.from_shape((3,))))
    b = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.int, st=ShapeTracker.from_shape((3,))))
    c = LazyOp(BinaryOps.ADD, src=(a,b))
    out0 = LazyOp(BufferOps.STORE, src=(LazyOp(BinaryOps.ADD, src=(a,c)),), arg=MemBuffer(idx=2, dtype=dtypes.int, st=ShapeTracker.from_shape((3,))))
    out1 = LazyOp(BufferOps.STORE, src=(LazyOp(BinaryOps.MUL, src=(a,b)),), arg=MemBuffer(idx=3, dtype=dtypes.int, st=ShapeTracker.from_shape((3,))))
    out2 = LazyOp(BufferOps.STORE, src=(LazyOp(ReduceOps.SUM, src=(a,)),), arg=MemBuffer(idx=4, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    out3 = LazyOp(BufferOps.STORE, src=(LazyOp(ReduceOps.MAX, src=(b,)),), arg=MemBuffer(idx=5, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    graph = create_graph([out0, out1, out2, out3])
    uops = MiniLinearizer(graph).linearize()
    alloc_data, init_outputs, prg, get_outputs = verify(uops)
    a, b = alloc_data([4,3,4]), alloc_data([10,20,3])
    outs = init_outputs([1,1,1,1])
    prg(a, b, *outs, global_size=(1,1,1), local_size=(1,1,1))
    assert get_outputs(outs) == [[18], [40], [11], [20]]

  def test_multi_output_reduce_alu(self):
    a = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=0, dtype=dtypes.int, st=ShapeTracker.from_shape((4,))))
    b = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    c = LazyOp(ReduceOps.SUM, src=(a,))
    out0 = LazyOp(BufferOps.STORE, src=(LazyOp(BinaryOps.ADD, src=(c,b)),), arg=MemBuffer(idx=2, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    out1 = LazyOp(BufferOps.STORE, src=(c,), arg=MemBuffer(idx=3, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    graph = create_graph([out0, out1])
    uops = MiniLinearizer(graph).linearize()
    alloc_data, init_outputs, prg, get_outputs = verify(uops)
    a, b = alloc_data([1,1,1,1]), alloc_data([2])
    outs = init_outputs([1,1])
    prg(a, b, *outs, global_size=(1,1,1), local_size=(1,1,1))
    assert get_outputs(outs) == [[6], [4]]

  def test_multi_dim_reduce(self):
    a = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=0, dtype=dtypes.int, st=ShapeTracker.from_shape((4,4))))
    b = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    reduce = LazyOp(ReduceOps.SUM, src=(a,), arg=((1,1)))
    out0 = LazyOp(BufferOps.STORE, src=(reduce,), arg=MemBuffer(idx=2, dtype=dtypes.int, st=ShapeTracker.from_shape((1,1))))
    out1 = LazyOp(BufferOps.STORE, src=(LazyOp(BinaryOps.MUL, src=(b,reduce), arg=((1,1))),), arg=MemBuffer(idx=3, dtype=dtypes.int, st=ShapeTracker.from_shape((1,1))))
    graph = create_graph([out0, out1])
    uops = MiniLinearizer(graph).linearize()
    alloc_data, init_outputs, prg, get_outputs = verify(uops)
    a, b = alloc_data(list(range(16))), alloc_data([2])
    outs = init_outputs([1,1])
    prg(a, b, *outs, global_size=(1,1,1), local_size=(1,1,1))
    assert get_outputs(outs) == [[120], [240]]

  def test_const_load_combo(self):
    a = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker.from_shape((1,))))
    b = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker.from_shape((1,))))
    c = LazyOp(BufferOps.CONST, src=(), arg=ConstBuffer(val=4.0, dtype=dtypes.float, st=ShapeTracker.from_shape((1,))))
    out0 = LazyOp(BufferOps.STORE, src=(LazyOp(BinaryOps.ADD, src=(a,b)),), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker.from_shape((1,))))
    out1 = LazyOp(BufferOps.STORE, src=(LazyOp(BinaryOps.MUL, src=(a,c)),), arg=MemBuffer(idx=3, dtype=dtypes.float, st=ShapeTracker.from_shape((1,))))
    uops = MiniLinearizer(create_graph([out0, out1])).linearize()
    alloc_data, init_outputs, prg, get_outputs = verify(uops)
    a, b = alloc_data([f32_to_bits(1)]), alloc_data([f32_to_bits(2)])
    outs = init_outputs([1,1])
    prg(a, b, *outs, global_size=(1,1,1), local_size=(1,1,1))

    assert get_outputs(outs, "f") == [[3.0], [4.0]]
