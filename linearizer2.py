import unittest, math
import numpy as np
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Any, Dict, List, Tuple, cast
from tinygrad.codegen.linearizer import expand_node
from tinygrad.codegen.uops import UOp, UOps
from tinygrad.device import Buffer, BufferCopy, Device, JITRunner
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import BinaryOps, BufferOps, ConstBuffer, LazyOp, LoadOps, MemBuffer, Op, ReduceOps
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor
from tinygrad.helpers import DEBUG, getenv

class ASTRunner(JITRunner):
  def __init__(self, ast: Tuple[LazyOp,...]):
    self.ast = ast
    self.device, self.compiler = Device[Device.DEFAULT], Device[Device.DEFAULT].compiler
    super().__init__()
  def __call__(self, rawbufs: List[Buffer], var_vals, wait=False, jit=False):
    assert self.compiler
    lin = MiniLinearizer(self.ast)
    lin.linearize()
    code = self.compiler.render("new_linearizer", lin.uops)
    if DEBUG >= 4: print(code)
    lib = self.compiler.compile(code)
    prg = self.device.runtime("new_linearizer", lib)
    prg(*[x._buf for x in rawbufs])

@dataclass(frozen=True)
class MiniScheduleItem:
  runner: JITRunner
  rawbufs: List[Buffer]

class Scheduler:
  def __init__(self) -> None:
    self.sched: List[MiniScheduleItem] = []
    self.rawbufs: List[Buffer] = []
    self.copy_cache: Dict[Buffer, int] = {}
    self.ast: TopologicalSorter = TopologicalSorter()

  def store_output(self, lb: LazyBuffer, src:LazyOp):
    op = LazyOp(BufferOps.STORE, (src, ), MemBuffer(len(self.rawbufs), lb.dtype, lb.st.simplify().unbind()[0]))
    store_buf = Buffer(lb.device, lb.size, lb.dtype)
    lb.base.realized = store_buf
    self.rawbufs.append(store_buf)
    self.ast.add(op, src)

  def create_schedule(self, lazy_buffers:List[LazyBuffer]):
    for lb in lazy_buffers:
      src = self._recursive_lazyop(lb, out_shape=lb.shape)
      self.store_output(lb, src)
    self.sched.append(MiniScheduleItem(ASTRunner(tuple(self.ast.static_order())), self.rawbufs))

  def _recursive_lazyop(self, lb:LazyBuffer, out_shape) -> LazyOp:
    # LoadOps have special sources
    if lb.base.op == LoadOps.CONST: # Consts are always generated
      return LazyOp(BufferOps.CONST, src=(), arg=ConstBuffer(val=lb.base.arg, dtype=lb.base.dtype, st=lb.st.simplify().unbind()[0]))
    if lb.base.op == LoadOps.COPY:
      host_buf = cast(Buffer,lb.base.srcs[0].realized)

      if host_buf in self.copy_cache:
        idx = self.copy_cache[host_buf]
        device_buf = self.rawbufs[idx]
      else:
        device_buf = Buffer(lb.device, lb.size, lb.dtype)
        self.sched.append(MiniScheduleItem(BufferCopy(), [device_buf, host_buf]))
        idx = len(self.rawbufs)
        self.copy_cache[host_buf] = len(self.rawbufs)
        self.rawbufs.append(device_buf)

      unbound_st, st_var_vals = lb.st.simplify().unbind()
      assert st_var_vals == {}, "variables not supported yet"
      return LazyOp(BufferOps.LOAD, (), MemBuffer(idx, lb.dtype, unbound_st))

    srcs = tuple([self._recursive_lazyop(src, out_shape) for src in lb.base.srcs])
    op = LazyOp(cast(Op,lb.base.op), src=srcs)
    self.ast.add(op, *srcs)
    return op

class MiniLinearizer:
  def __init__(self, ast):
    self.ast = ast
    self.uops: List[UOp] = []
    self.buf_pointers: Dict[int, UOp] = {}
    self.loaded_bufs: Dict[MemBuffer, UOp] = {}
    self.alu_cache: Dict[Any, UOp] = {}
    self.loop_uops = {}

  def const(self, val, dtype=dtypes.int):
    existing = [u for u in self.uops if u.uop == UOps.CONST and u.arg == val]
    if len(existing) != 0: return existing[0]
    uop = UOp(UOps.CONST, dtype=dtype, arg=val)
    self.uops.append(uop)
    return uop

  def get_reduce_acc(self, dtype, op):
    if op == ReduceOps.SUM: return 0.0 if dtypes.is_float(dtype) else 0
    if op == ReduceOps.MAX:
      if dtypes.is_int(dtype): return 0 if dtypes.is_unsigned(dtype) else -2**(dtype.itemsize*8-1)
      return -math.inf if dtypes.is_float(dtype) else False

  def _lower_op(self, op:LazyOp) -> UOp:
    if op.op == BufferOps.LOAD: return self.loaded_bufs[op.arg]
    if op.op == BufferOps.CONST: return self.const(op.arg.val, op.arg.dtype)
    if op.op in ReduceOps:
      buf: MemBuffer = op.src[0].arg
      acc = UOp(UOps.DEFINE_ACC, dtype=buf.dtype, arg=self.get_reduce_acc(buf.dtype,op.op))
      e_idx = buf.st.expr_idxs()[0]
      assert isinstance(e_idx, Variable)
      assert len(expand_node(e_idx)) == 1, "todo!"

      if e_idx.expr in self.loop_uops: loop = self.loop_uops[e_idx.expr]
      else: loop = UOp(UOps.LOOP, dtype=dtypes.int, vin=(self.const(e_idx.min),self.const(e_idx.max+1)))

      src = UOp(UOps.LOAD, dtype=buf.dtype, vin=(self.buf_pointers[buf.idx],loop))
      alu = UOp(UOps.ALU, dtype=src.dtype, vin=(acc,src), arg=BinaryOps.ADD if op.op == ReduceOps.SUM else BinaryOps.MAX)
      ret = UOp(UOps.PHI, dtype=src.dtype, vin=(acc,alu,loop))

      if e_idx.expr in self.loop_uops:
        idx = self.uops.index(loop)
        self.uops = self.uops[:idx] + [acc, self.uops[idx]] + [src, alu, ret] + self.uops[idx+1:]
      else:
        self.uops += [acc, loop, src, alu, ret, UOp(UOps.ENDLOOP, vin=(loop,))]
      self.loop_uops[e_idx.expr] = loop
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
        self.buf_pointers[buf.idx] = UOp(UOps.DEFINE_GLOBAL, dtype=PtrDType(buf.dtype), arg=f"data{buf.idx}")
        self.uops.append(self.buf_pointers[buf.idx])
      if op.op == BufferOps.LOAD and buf not in self.loaded_bufs:
        self.loaded_bufs[buf] = UOp(UOps.LOAD, dtype=buf.dtype, vin=(self.buf_pointers[buf.idx],self.const(0)))
        self.uops.append(self.loaded_bufs[buf])
      else:
        ret = self._lower_op(op.src[0])
        self.uops.append(UOp(UOps.STORE, dtype=ret.dtype, vin=(self.buf_pointers[buf.idx],self.const(0),ret)))
    return self.uops

class TestLinearizer2(unittest.TestCase):
  assert getenv("LINEARIZER2"), "please use LINEARIZER2=1 to render the bufs correctly in cstyle"
  def _new_realize(self, vals):
    scheduler = Scheduler()
    scheduler.create_schedule([x.lazydata for x in vals])
    for si in scheduler.sched:
      si.runner(si.rawbufs, var_vals={})
    ret = [np.frombuffer(x.lazydata.base.realized.as_buffer(), dtype=x.dtype.np).reshape(x.shape) for x in vals]
    for x in vals: x.lazydata.base.realized = None # reset values for the comparison
    return ret

  def test_multi_output_simple(self):
    a = Tensor([2])
    b = Tensor([6])
    out0 = a - b
    out1 = a * b
    outputs = [out0, out1]

    ret = self._new_realize(outputs)
    expected = [x.numpy() for x in outputs]
    np.testing.assert_equal(ret, expected)

  def test_multi_output_multi_reduce(self):
    a = Tensor([1,2,3,4])
    b = Tensor([22])
    out0 = a.sum()
    out1 = out0 + b
    out2 = a.max()
    outputs = [out0, out1, out2]

    ret = self._new_realize(outputs)
    expected = [x.numpy() for x in outputs]
    np.testing.assert_equal(ret, expected)

  @unittest.skip("TODO - needs a good scheduler infra, reconsidering if this should be possible")
  def test_multi_dim_reduce(self):
    # even though doing two reduces on different shapes might not be profitable, we should be able to linearize it
    a = Tensor([[2,2], [3,3]])
    b = Tensor([1,2,3])
    out0 = a.sum()
    out1 = b.sum()
    outputs = [out0, out1]

    ret = self._new_realize(outputs)
    expected = [x.numpy() for x in outputs]
    np.testing.assert_equal(ret, expected)

  @unittest.skip("TODO - needs a good scheduler infra")
  def test_batchnorm(self):
    x_data = np.random.randn(2, 4, 3, 3)
    weight_data, bias_data = np.ones(4), np.zeros(4)
    x, weight, bias = [Tensor(data, dtype=dtypes.float) for data in [x_data,weight_data,bias_data]]
    batch_mean = x.mean(axis=(0,2,3))
    y = (x - batch_mean.reshape(shape=[1, -1, 1, 1]))
    batch_var = (y*y).mean(axis=(0,2,3))
    batch_invstd = batch_var.add(1e-5).pow(-0.5)
    out = x.batchnorm(weight, bias, batch_mean, batch_invstd)

    scheduler = Scheduler()
    scheduler.create_schedule([cast(LazyBuffer,out.lazydata)])
    """
    #ast = scheduler.sched[-1].runner.ast
    #for si in scheduler.sched: print(si)
    for op in ast:
      if op.op == BufferOps.LOAD:
        print(op.op)
    print(len(scheduler.rawbufs))
    """
