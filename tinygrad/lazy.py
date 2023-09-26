from __future__ import annotations
import math
from typing import Optional, Union, cast, Tuple, Any, List, Dict, Mapping, Callable
from tinygrad.shape.shapetracker import ShapeTracker
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

OPT = getenv("OPT", 2)
LAZYCACHE = getenv("LAZYCACHE", 1)

MERGE_ELEMENTWISE_INTO_REDUCE, MERGE_ELEMENTWISE_OPS, SHUFFLE_MOVEMENT_OPS = OPT>=1, OPT>=1, OPT>=1
MERGE_ONE_REDUCE_INTO_ELEMENTWISE, SHUFFLE_PAD_OPS = OPT>=2, OPT>=2
PUSH_PERMUTES = OPT>=3  # TODO: reimplement this
UNSAFE_PAD_OPS = {BinaryOps.DIV, BinaryOps.CMPLT, UnaryOps.LOG2, UnaryOps.EXP2, UnaryOps.RECIP}

def fix_unbased_shape(src:LazyBuffer) -> LazyOp:
  assert src.is_contiguous(), "non contiguous can't be fixed"
  # if it's contiguous, applying a reshape to the base is safe
  return src.base.op.map_buffers({x:x.reshape(src.shape) for x in src.base.op.buffers}) if src.base != src else src.op

def _ast_reduceops(op:LazyOp) -> LazyOp:
  # TODO: this can also corealize a binary op after the reduce, not just before
  src = op.src[0]
  if not src.realized:
    assert isinstance(src, LazyBuffer), "src must be a LazyBuffer"
    assert isinstance(src.base.op, LazyOp), "if not src.realized, then src.op must be a LazyOp"
    if MERGE_ELEMENTWISE_INTO_REDUCE and src.base.op.op in ElementwiseOps and len(src.children) <= 1 and src.is_contiguous():
      # it's contiguous, but it might require reshapes of the binaryops
      src = fix_unbased_shape(src)
  return LazyOp(op.op, (src,), op.arg)

# this supports late merging an upstream Reduce op and even an Elementwise op above that
def _ast_binaryops(op:LazyOp, output_shape:Tuple[sint, ...]) -> LazyOp:
  real_srcs: Dict[LazyBuffer, Optional[Union[LazyOp, LazyBuffer]]] = {x:None for x in op.buffers}
  # NOTE: contiguous does not always mean the same size with SHRINK. this is still mergeable but requires more thought how
  # TODO: this can also support late fusion of BinaryOps, required for test_fold_conv_sgd
  psrcs: List[Tuple[LazyBuffer, LazyBuffer]] = [(k,x) for k,x in zip(real_srcs.keys(), [x.base for x in real_srcs.keys()]) if not x.realized and k.is_contiguous() and x.op.op in ReduceOps and len(x.children) <= 1]

  intermediate_shape: Tuple[sint, ...] = output_shape
  if MERGE_ONE_REDUCE_INTO_ELEMENTWISE and psrcs:
    psrc = psrcs[0] # NOTE: right now we can't handle multiple, as we'd have to check for loop
    if psrc[1].op.op in ReduceOps:
      top = _ast_reduceops(psrc[1].op)
    real_srcs[psrc[0]] = top
    real_srcs.update({x:x for x in top.buffers})  # the reduce op buffers are not modified

    # if the ReduceOp is followed by a reshape, we push this reshape before all the ElementwiseOp inputs
    if psrc[0].shape != psrc[1].shape:
      intermediate_shape = psrc[1].shape
      assert psrc[0].shape == output_shape, f"shape mismatch {psrc[0].shape} != {output_shape}"

  # reshape all the late ops into the output shape
  # NOTE: these RESHAPEs will return self if they don't change the shape
  for x in real_srcs.keys():
    if real_srcs[x] is None: real_srcs[x] = x.reshape(intermediate_shape)

  # NOTE: cast the type to remove the Optional
  return op.map_buffers(cast(Dict[LazyBuffer, Union[LazyOp, LazyBuffer]], real_srcs))

def _ast_bufferops(op:LazyOp) -> LazyOp:
  replacements:Dict[LazyBuffer, LazyOp] = {}
  base_bufs = dedup([x.base for x in op.buffers if x.realized or not isinstance(Device[x.device], Compiled) or x.device == "LLVM" or x.base.op.op != LoadOps.CONST])
  for x in op.buffers:
    if x.base in base_bufs:
      if x.realized and isinstance(x.realized, RawConst):
        replacements[x] = LazyOp(BufferOps.CONST, (), ConstBuffer(float(x.realized._buf), x.dtype, x.st.simplify()))
      else:
        replacements[x] = LazyOp(BufferOps.MEM, (), MemBuffer(base_bufs.index(x.base)+1, x.dtype, x.st.simplify()))
    elif x.base.op.op == LoadOps.CONST:
      replacements[x] = LazyOp(BufferOps.CONST, (), ConstBuffer(float(x.base.op.arg), x.dtype, x.st.simplify()))
    else:
      raise NotImplementedError(f"not handled {x}")
  return op.map_buffers(replacements)

class LazyBuffer:
  def __init__(self, op:Optional[LazyOp], st:ShapeTracker, dtype:DType, device:str, src:Optional[RawBuffer]=None, base:Optional[LazyBuffer]=None):
    self.st: ShapeTracker = st
    self.shape, self.dtype, self.device = self.st.shape, dtype, device
    self.output_buffer: Optional[RawBuffer] = None
    self.var_vals: Dict = {}
    if base:
      assert base.st.contiguous, "base must be contiguous"
      self._base: LazyBuffer = base
    else:
      self._realized: Optional[RawBuffer] = src
      self._children: WeakSet = WeakSet()
      if op:
        self.op: LazyOp = op
        for x in op.buffers: x.children.add(self)

  lazycache: WeakValueDictionary = WeakValueDictionary()
  @staticmethod
  def cache(op:Optional[LazyOp], st:ShapeTracker, dtype:DType, device:str, base:Optional[LazyBuffer]=None):
    if not LAZYCACHE: return LazyBuffer(op, st, dtype, device, base=base)
    wop = (ref(op) if op else None, st, dtype, device, ref(base) if base else None)
    if wop in LazyBuffer.lazycache:
      if op:
        for x in op.buffers: x.children.add(LazyBuffer.lazycache[wop])
      return LazyBuffer.lazycache[wop]
    LazyBuffer.lazycache[wop] = ret = LazyBuffer(op, st, dtype, device, base=base)
    return ret

  def __repr__(self): return f"<L{'B' if self.base == self else 'V'} {self.shape} {self.dtype} op={(self.op.op if hasattr(self, 'op') else '') if not self.realized else self.realized} st={self.st}>"
  def _device_extra_args(self) -> Dict[str, str]: return {"device": self.device.split(":", 1)[1]} if ":" in self.device else {}
  def is_contiguous(self): return self.st.contiguous and self.base.st.size() == self.st.size()

  # handle base
  @property
  def base(self): return self._base if hasattr(self, '_base') else self
  @property
  def realized(self): return self.base._realized
  @realized.setter
  def realized(self, val):
    assert self.base == self, "must be a base"
    self._realized = val
  @property
  def children(self): return self.base._children

  def contiguous(self) -> LazyBuffer:
    if self.is_contiguous(): return self
    return LazyBuffer.cache(LazyOp(LoadOps.CONTIGUOUS, (self,)), ShapeTracker.from_shape(self.shape), self.dtype, self.device)

  @staticmethod
  def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
    return LazyBuffer(LazyOp(op, tuple() if src is None else (src,), arg), ShapeTracker.from_shape(tuple(shape)), dtype, device)

  def const(self, val:Union[float, int]) -> LazyBuffer:
    return self.loadop(LoadOps.CONST, tuple(), dtypes.from_np(self.dtype.np), self.device, arg=val).reshape((1,)*len(self.shape)).expand(self.shape)

  @staticmethod
  def fromCPU(x: np.ndarray) -> LazyBuffer:
    return LazyBuffer(None, ShapeTracker.from_shape(x.shape), dtypes.from_np(x.dtype), "CPU", src=RawNumpyBuffer.fromCPU(x))

  def toCPU(self) -> np.ndarray:
    assert self.dtype.np, f"{self.dtype} is not supported in toCPU"
    self_casted = self.e(UnaryOps.CAST, arg=(dtypes.from_np(self.dtype.np), False)) if dtypes.from_np(self.dtype.np) != self.dtype else self
    realized = self_casted.contiguous().realize().realized
    assert all_int(self.shape), f"no toCPU if shape is symbolic, {self.shape=}"
    return cast(RawBuffer, realized).toCPU().reshape(self.shape)

  # *** elementwise ops ***

  def e(self:LazyBuffer, op:Union[UnaryOps, BinaryOps, TernaryOps], *srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
    srcs = (self,)+srcs
    out_dtype = max([x.dtype for x in srcs]) if op != UnaryOps.CAST else cast(Tuple[DType, bool], arg)[0]

    if MERGE_ELEMENTWISE_OPS:
      # remove the buffers from any (childless) BinaryOps that feed into this
      srcs = tuple([fix_unbased_shape(x) if not x.realized and x.is_contiguous() and x.base.op.op in ElementwiseOps and not x.children else x for x in srcs])  # type: ignore

    return LazyBuffer.cache(LazyOp(op, srcs, arg), ShapeTracker.from_shape(self.shape), out_dtype, self.device)

  # *** reduce ops ***

  def _reduce_op(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[sint, ...]) -> LazyBuffer:
    if self.shape == tuple(new_shape): return self
    return LazyBuffer.cache(LazyOp(op, (self,), new_shape), ShapeTracker.from_shape(new_shape), self.dtype, self.device)

  def r(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[sint, ...]) -> LazyBuffer:
    if any(not isinstance(s, int) for s in self.shape) or prod(self.shape) // prod(new_shape) < 32768: return self._reduce_op(op, new_shape) # The amount of work should be big enough to take the benefit of "2 kernels" approach.
    heuristic, divisor, dim_to_split = max(((divisor := math.gcd(256, old))/(stride or math.inf), divisor, i) for i, (old, new, stride) in enumerate(zip(self.shape, new_shape, self.st.real_strides())) if old != new) # type: ignore
    if divisor < 16 or heuristic < 0.1: return self._reduce_op(op, new_shape) # Choose largest divisor (>=16) to split on, penalize large strides.
    def splitted_shape(dim_aft_div): return self.shape[:dim_to_split] + (self.shape[dim_to_split]//divisor,) + dim_aft_div + self.shape[dim_to_split+1:]
    return self.reshape(splitted_shape((divisor,)))._reduce_op(op, splitted_shape((1,))).reshape(splitted_shape(()))._reduce_op(op, new_shape)

  # *** movement ops ***

  def _movement_op(self, fxn, arg, push_ewop=False, unsafe_ops=None) -> LazyBuffer:
    st:ShapeTracker = fxn(self.st, arg)
    if self.st == st: return self
    if push_ewop and not self.realized and self.base == self and self.op.op in ElementwiseOps:
      if not unsafe_ops or not any(x.op in UNSAFE_PAD_OPS for x in self.op.get_lazyops()):
        mapped = self.op.map_buffers({x:x._movement_op(fxn, arg) for x in self.op.buffers})
        return LazyBuffer.cache(mapped, ShapeTracker.from_shape(st.shape), self.dtype, self.device)
    return LazyBuffer.cache(None, st, self.dtype, self.device, base=self.base)

  def permute(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.permute, arg, SHUFFLE_MOVEMENT_OPS)
  def shrink(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.shrink, arg, SHUFFLE_MOVEMENT_OPS)
  def stride(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.stride, arg, SHUFFLE_MOVEMENT_OPS)
  def pad(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.pad, arg, SHUFFLE_PAD_OPS, UNSAFE_PAD_OPS)
  def reshape(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.reshape, arg)
  def expand(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.expand, arg)

  @property
  def buffers(self) -> Tuple[LazyBuffer, ...]: return (self,)
  def map_buffers(self, real_srcs: Mapping[LazyBuffer, Union[LazyBuffer, LazyOp]]): return real_srcs.get(self, self)
  def get_lazyops(self) -> List[LazyOp]: return []

  def schedule(self:LazyBuffer, seen=None) -> List[Tuple[LazyOp, List[LazyBuffer]]]:
    if seen is None: seen = set()
    if self in seen: return []
    seen.add(self)
    if self.realized: return []
    if self.base != self: return self.base.schedule(seen)
    # NOTE: late rewrite contiguous
    op = self.op if self.op.op != LoadOps.CONTIGUOUS else LazyOp(UnaryOps.NOOP, self.op.src)
    if op.op in LoadOps:
      return [(self.op, [self])]
    if op.op in ElementwiseOps:
      op = _ast_binaryops(op, self.shape)
    elif op.op in ReduceOps:
      op = _ast_reduceops(op)
    buffers = list(op.buffers)
    op = _ast_bufferops(op)
    ret = []
    for x in buffers:
      if not x.realized:
        for _op,_buffers in x.schedule(seen):
          ret.append((_op,_buffers))
    return ret+[(op, [self]+buffers)]

  def realize(self:LazyBuffer) -> LazyBuffer:
    if not self.realized:
      for op,buffers in self.schedule():
        if DEBUG >= 3:
          from extra.utils import print_tree
          print_tree(op)
        if op.op in LoadOps:
          LOAD_OPS_DISPATCHER[cast(LoadOps, op.op)](buffers[0])
        else:
          realized_bufs = dedup([x.realized for x in buffers[1:] if buf_is_kernel_arg(x)])
          buffers[0].realized = Device[buffers[0].device].exec_ast(op, output=buffers[0], inputs=realized_bufs, var_vals={}, **self._device_extra_args())
          del buffers[0].op
    return self

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