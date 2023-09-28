from __future__ import annotations

import math, functools
from typing import Optional, Union, cast, Tuple, Any, List, Dict, Mapping, Callable
from tinygrad.shape.shapetracker import ShapeTracker, get_common_shape, reduce_expand, reduce_contract_arg, get_contraction
from tinygrad.ops import LazyOp, LoadOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps
ElementwiseOps = {*UnaryOps, *BinaryOps, *TernaryOps}

#from tinygrad.graph import log_op
from tinygrad.ops import BufferOps, ConstBuffer, MemBuffer, Device, Compiled
from tinygrad.helpers import DType, dtypes, all_int, dedup, DEBUG, getenv, prod
from tinygrad.runtime.lib import RawConst, RawBuffer, buf_is_kernel_arg
from tinygrad.runtime.ops_cpu import RawNumpyBuffer
from tinygrad.shape.symbolic import sint
from weakref import WeakSet, WeakValueDictionary, ref
import numpy as np
from tinygrad.graph import log_op

OPT = getenv("OPT", 2)

MERGE_ELEMENTWISE_INTO_REDUCE, MERGE_ELEMENTWISE_OPS, SHUFFLE_MOVEMENT_OPS = OPT>=1, OPT>=1, OPT>=1
MERGE_ONE_REDUCE_INTO_ELEMENTWISE = OPT>=2
#UNSAFE_PAD_OPS = {BinaryOps.DIV, BinaryOps.CMPLT, UnaryOps.LOG2, UnaryOps.EXP2, UnaryOps.RECIP}

class LazyCommon:
  @property
  def buffers(self) -> Tuple[LazyBuffer, ...]: return (self,)
  def map_buffers(self, real_srcs: Mapping[LazyBuffer, Union[LazyBuffer, LazyOp]]): return real_srcs[self].map_buffers(real_srcs) if self in real_srcs and real_srcs[self] != self else self
  def get_lazyops(self) -> List[LazyOp]: return []

# this is a contiguous array and can have an op fill it
class LazyBacking(LazyCommon):
  def __init__(self, op:Optional[LazyOp], size:int, dtype:DType, device:str, src:Optional[RawBuffer]=None):
    self.size, self.dtype, self.device = size, dtype, device
    self.realized: Optional[RawBuffer] = src
    self.output_buffer: Optional[RawBuffer] = None
    self.children: WeakSet = WeakSet()
    self.siblings: WeakSet = WeakSet()
    if op:
      self.op: LazyOp = op
      if DEBUG >= 6: print(f"create LazyBacking {self}")
      for x in op.buffers:
        x.children.add(self)
        if DEBUG >= 6: print(f"   adding to children of {x}")
  def __repr__(self): return f"<LB {id(self)} {self.size} {self.dtype} op={(self.op.op if hasattr(self, 'op') else '') if not self.realized else self.realized}>"

  lazycache: WeakValueDictionary = WeakValueDictionary()
  @staticmethod
  def cache(op:LazyOp, size:int, dtype:DType, device:str):
    wop = (ref(op), size, dtype, device)
    if wop in LazyBacking.lazycache: return LazyBacking.lazycache[wop]
    LazyBacking.lazycache[wop] = ret = LazyBacking(op, size, dtype, device)
    return ret

  def _device_extra_args(self) -> Dict[str, str]: return {"device": self.device.split(":", 1)[1]} if ":" in self.device else {}

  @property
  def base(self) -> LazyBacking: return self
  @property
  def shape(self): return (self.size,)
  @property
  def st(self): return ShapeTracker.from_shape((self.size,))

  def toCPU(self) -> np.ndarray:
    assert self.dtype.np, f"{self.dtype} is not supported in toCPU"
    #self_casted = self.e(UnaryOps.CAST, arg=(dtypes.from_np(self.dtype.np), False)) if dtypes.from_np(self.dtype.np) != self.dtype else self
    return self.realize().realized.toCPU()

  def schedule(self, seen=None):
    if seen is None: seen = set()
    if self in seen or self.realized: return []
    seen.add(self)

    op = self.op if self.op.op != LoadOps.CONTIGUOUS else LazyOp(UnaryOps.NOOP, self.op.src)
    if op.op in LoadOps: return [(self.op, self, ())]

    # fusion.
    # in fusion, we start with an op
    # ops have srcs, some of which we replace with other ops
    # we are creating a dictionary of those replacements
    replacements: Dict[LazyBacking, LazyOp] = {self:op}

    def elementwise_fusion(op:LazyOp):
      assert op.op in ElementwiseOps
      for src in op.src:
        if not src.realized and isinstance(src, LazyBacking) and all(x in replacements for x in src.children):
          if MERGE_ELEMENTWISE_OPS and src.base.op.op in ElementwiseOps and src.st.size() <= src.base.size:
            replacements[src] = src.op
            elementwise_fusion(src.op)

    def reduceop_fusion(op:LazyOp):
      assert op.op in ReduceOps
      src = op.src[0]
      if MERGE_ELEMENTWISE_INTO_REDUCE and not src.realized and isinstance(src, LazyBacking) and src.op.op in ElementwiseOps and all(x in replacements for x in src.children):
        replacements[src] = src.op
        elementwise_fusion(src.op)

    if op.op in ElementwiseOps:
      elementwise_fusion(op)
      if MERGE_ONE_REDUCE_INTO_ELEMENTWISE:  # we can also fuse a (single) reduceop
        op = op.map_buffers(replacements)
        for src in op.buffers:
          if isinstance(src, LazyBacking) and not src.realized and src.op.op in ReduceOps and all(x in replacements for x in src.children):
            replacements[src] = src.op
            reduceop_fusion(src.op)
            break
    elif op.op in ReduceOps:
      reduceop_fusion(op)
    op = op.map_buffers(replacements)

    base_bufs = dedup([x.base for x in op.buffers if x.realized or not isinstance(Device[x.device], Compiled) or x.device == "LLVM" or x.base.op.op != LoadOps.CONST])
    ret = []
    for x in base_bufs: ret += x.schedule(seen)

    reduceops: List[LazyOp] = dedup([x for x in op.get_lazyops() if x.op in ReduceOps])
    assert len(reduceops) <= 1, "max one reduce op in an ast"

    # get buffer shapes
    if reduceops:
      late_shapes = [x.shape for x in op.buffers if x not in reduceops[0].buffers]
      tmp_early_shape = get_common_shape([x.shape for x in reduceops[0].buffers] + [reduceops[0].arg[0]])
      reduced_shape = reduce_expand(tmp_early_shape, reduceops[0].arg[0], reduceops[0].arg[1])
      late_shape = get_common_shape(late_shapes + [reduceops[0].arg[1], reduced_shape])
      early_shape = []
      for es, c in zip(tmp_early_shape, get_contraction(late_shape, reduced_shape)):
        if len(c) > 1:
          piece_of_late_shape = [late_shape[x] for x in c]
          assert prod(piece_of_late_shape) == es, f"late shape {piece_of_late_shape} early piece {es}"
          early_shape += piece_of_late_shape
        else:
          early_shape.append(es)
      early_shape = tuple(early_shape)
      assert len(early_shape) == len(late_shape), f"shape len mismatch {early_shape} != {late_shape}"
    else:
      late_shape = get_common_shape([x.shape for x in op.buffers])

    # replace the buffers with BufferOps with the correct shape
    # uses: op.buffers, base_bufs, and buffer_shapes
    replacements:Dict[LazyBuffer, LazyOp] = {}
    for x in op.buffers:
      st = x.st.reshape(early_shape if reduceops and x in reduceops[0].buffers else late_shape).simplify()
      if x.base in base_bufs:
        if x.realized and isinstance(x.realized, RawConst):
          replacements[x] = LazyOp(BufferOps.CONST, (), ConstBuffer(float(x.realized._buf), x.dtype, st))
        else:
          replacements[x] = LazyOp(BufferOps.MEM, (), MemBuffer(base_bufs.index(x.base)+1, x.dtype, st))
      elif x.base.op.op == LoadOps.CONST:
        replacements[x] = LazyOp(BufferOps.CONST, (), ConstBuffer(float(x.base.op.arg), x.dtype, st))
      else:
        raise NotImplementedError(f"not handled {x}")
    log_op(self, op)
    #print(f"schedule {self} with {len(self.children)} children")
    return ret + [(op.map_buffers(replacements), self, base_bufs)]

  def realize(self:LazyBuffer) -> LazyBuffer:
    if not self.realized:
      for op,out,buffers in self.schedule():
        if DEBUG >= 3:
          from extra.utils import print_tree
          print_tree(op)
        if op.op in LoadOps:
          LOAD_OPS_DISPATCHER[cast(LoadOps, op.op)](out)
        else:
          realized_bufs = dedup([x.realized for x in buffers if buf_is_kernel_arg(x)])
          out.realized = Device[out.device].exec_ast(op, output=out, inputs=realized_bufs, var_vals={}, **self._device_extra_args())
          del out.op
        assert out.realized.dtype == out.dtype, "realized dtype is correct"
    return self

# this is a view into a contiguous LazyBacking. this is always non-contiguous
class LazyView(LazyCommon):
  def __init__(self, st:ShapeTracker, base:LazyBacking):
    assert not st.contiguous or st.size() != base.size, "LazyView should never be contiguous"
    self.st: ShapeTracker = st
    self.base: LazyBacking = base
    #self.base.children.add(self)
    #self.children: WeakSet = WeakSet()
    if DEBUG >= 6: print(f"create LazyView {self}")

  def __repr__(self): return f"<LV {id(self)} {self.st.views} {self.base}>"

  lazycache: WeakValueDictionary = WeakValueDictionary()
  @staticmethod
  def cache(st:ShapeTracker, base:LazyBacking) -> LazyCommon:
    if st.contiguous and st.size() == base.size: return base
    st = st.canonical()
    wop = (st, ref(base))
    if wop in LazyView.lazycache: return LazyView.lazycache[wop]
    LazyView.lazycache[wop] = ret = LazyView(st, base)
    return ret

  @property
  def shape(self): return self.st.shape
  @property
  def dtype(self): return self.base.dtype
  @property
  def device(self): return self.base.device
  @property
  def realized(self): return self.base.realized
  @property
  def children(self): return self.base.children
  def schedule(self): return self.base.schedule()
  def realize(self): return self.base.realize()
  def toCPU(self) -> np.ndarray: return LazyBuffer(self.shape, self).contiguous().backing.toCPU()

# this is the normal LazyBuffer class. it has a shape and behaves like a reasonable tensor should
# there's no longer any point in caching it, as we just cache the underlying stuff
class LazyBuffer:
  def __init__(self, shape:Tuple[sint, ...], backing:Union[LazyBacking, LazyView]):
    assert isinstance(shape, tuple)
    self.shape, self.backing = shape, backing

  def __repr__(self): return f"<L {self.shape} {self.backing}>"

  # passthrough properties
  @property
  def device(self): return self.backing.device
  @property
  def dtype(self): return self.backing.dtype
  @property
  def realized(self): return self.backing.realized
  @property
  def st(self): return self.backing.st  # NOTE: the shape is wrong
  def schedule(self): return self.backing.schedule() #set())
  def realize(self): return self.backing.realize()

  def contiguous(self) -> LazyBuffer:
    if isinstance(self.backing, LazyBacking): return self
    return LazyBuffer(self.shape, LazyBacking(LazyOp(LoadOps.CONTIGUOUS, (self.backing,)), prod(self.shape), self.dtype, self.device))

  # *** external entrypoints ***

  @staticmethod
  def loadop(op, shape, dtype, device, arg=None, src:Optional[LazyBuffer]=None) -> LazyBuffer:
    return LazyBuffer(shape, LazyBacking(LazyOp(op, tuple() if src is None else (src.backing,), arg), prod(shape), dtype, device))

  def const(self, val:Union[float, int]) -> LazyBuffer:
    return self.loadop(LoadOps.CONST, tuple(), dtypes.from_np(self.dtype.np), self.device, arg=val).reshape((1,)*len(self.shape)).expand(self.shape)

  @staticmethod
  def fromCPU(x: np.ndarray) -> LazyBuffer:
    return LazyBuffer(x.shape, LazyBacking(None, prod(x.shape), dtypes.from_np(x.dtype), "CPU", src=RawNumpyBuffer.fromCPU(x)))

  def toCPU(self) -> np.ndarray: return self.backing.toCPU().reshape(self.shape)

  # *** elementwise ops ***

  def e(self:LazyBuffer, op:Union[UnaryOps, BinaryOps, TernaryOps], *srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
    bsrcs = tuple(x.backing for x in (self,)+srcs)
    out_dtype = max([x.dtype for x in bsrcs]) if op != UnaryOps.CAST else cast(Tuple[DType, bool], arg)[0]
    return LazyBuffer(self.shape, LazyBacking.cache(LazyOp(op, bsrcs, arg), prod(self.shape), out_dtype, self.device))

  # *** reduce ops ***

  def _reduce_op(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[sint, ...]) -> LazyBuffer:
    old_shape, small_new_shape = reduce_contract_arg(self.shape, new_shape)
    if old_shape == small_new_shape: return self
    return LazyBuffer(new_shape, LazyBacking.cache(LazyOp(op, (self.backing,), (old_shape, small_new_shape)), prod(new_shape), self.dtype, self.device))

  def r(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[sint, ...]) -> LazyBuffer:
    if any(not isinstance(s, int) for s in self.shape) or prod(self.shape) // prod(new_shape) < 32768: return self._reduce_op(op, new_shape) # The amount of work should be big enough to take the benefit of "2 kernels" approach.
    heuristic, divisor, dim_to_split = max(((divisor := math.gcd(256, old))/(stride or math.inf), divisor, i) for i, (old, new, stride) in enumerate(zip(self.shape, new_shape, self.backing.st.reshape(self.shape).real_strides())) if old != new) # type: ignore
    if divisor < 16 or heuristic < 0.1: return self._reduce_op(op, new_shape) # Choose largest divisor (>=16) to split on, penalize large strides.
    def splitted_shape(dim_aft_div): return self.shape[:dim_to_split] + (self.shape[dim_to_split]//divisor,) + dim_aft_div + self.shape[dim_to_split+1:]
    return self.reshape(splitted_shape((divisor,)))._reduce_op(op, splitted_shape((1,))).reshape(splitted_shape(()))._reduce_op(op, new_shape)

  # *** movement ops ***

  def _movement_op(self, fxn, arg, push_ewop=False) -> LazyBuffer:
    st:ShapeTracker = fxn(self.backing.st.reshape(self.shape), arg)
    if push_ewop and not self.realized and isinstance(self.backing, LazyBacking) and self.backing.op.op in ElementwiseOps:# and not self.backing.children:
      op = self.backing.op
      # self is no longer children of its buffers
      # NOTE: do children have to be refcounted
      for x in op.buffers:
        if self.backing in x.children: x.children.remove(self.backing)
      mapped = op.map_buffers({x:LazyBuffer(self.shape, x)._movement_op(fxn, arg, push_ewop).backing for x in op.buffers})
      return LazyBuffer(st.shape, LazyBacking.cache(mapped, prod(st.shape), self.dtype, self.device))
    return LazyBuffer(st.shape, LazyView.cache(st, self.backing.base))

  def permute(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.permute, arg, SHUFFLE_MOVEMENT_OPS)
  def shrink(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.shrink, arg, SHUFFLE_MOVEMENT_OPS)
  def stride(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.stride, arg, SHUFFLE_MOVEMENT_OPS)
  def pad(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.pad, arg)
  def reshape(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.reshape, arg)
  def expand(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.expand, arg)

  def map_buffers(self, real_srcs: Mapping[LazyBuffer, Union[LazyBuffer, LazyOp]]): return real_srcs.get(self, self)
  def get_lazyops(self) -> List[LazyOp]: return []

# *** loadop realization (unrelated to lazy) ***

def _realize_from(buffer: LazyBuffer) -> None:
  rawbuf = buffer.op.src[0].realize()
  assert rawbuf.realized, "realize failed?"
  if DEBUG >= 3: print(f"*** copy {buffer.device} <- {rawbuf.device} size {rawbuf.realized.size} dtype {rawbuf.realized.dtype}")
  buffer.realized = Device[buffer.device].buffer.fromCPU(rawbuf.toCPU(), **buffer._device_extra_args())

def _realize_rand(buffer: LazyBuffer) -> None:
  rng = np.random.default_rng(buffer.op.arg)
  buffer.realized = Device[buffer.device].buffer.fromCPU(rng.random(size=buffer.shape, dtype=np.float32).astype(dtype=buffer.dtype.np, copy=False), **buffer._device_extra_args()) # type: ignore

def _realize_const(buffer: LazyBuffer) -> None:
  if isinstance(Device[buffer.device], Compiled) and buffer.device not in ["LLVM"]:  # consts are broken in LLVM in NaN/inf
    buffer.realized = RawConst(1, buffer.dtype, float(buffer.op.arg))
  else:
    buffer.realized = Device[buffer.device].buffer.fromCPU(np.array(buffer.op.arg, dtype=buffer.dtype.np), **buffer._device_extra_args())

def _realize_empty(buffer: LazyBuffer) -> None:
  assert all_int(buffer.shape), "does not support symbolic shape"
  buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())

def _realize_custom(buffer: LazyBuffer) -> None:
  # this needs to immediately realize
  buffer.realized = buffer.op.arg(buffer, *[x.realize() for x in buffer.op.src])

LOAD_OPS_DISPATCHER: Dict[LoadOps, Callable] = {
  LoadOps.CUSTOM: _realize_custom,
  LoadOps.FROM: _realize_from,
  LoadOps.EMPTY: _realize_empty,
  LoadOps.CONST: _realize_const,
  LoadOps.RAND: _realize_rand,
}