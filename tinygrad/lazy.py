from __future__ import annotations

import math
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
LAZYCACHE = getenv("LAZYCACHE", 1)

MERGE_ELEMENTWISE_INTO_REDUCE, MERGE_ELEMENTWISE_OPS, SHUFFLE_MOVEMENT_OPS = OPT>=1, OPT>=1, OPT>=1
MERGE_ONE_REDUCE_INTO_ELEMENTWISE, SHUFFLE_PAD_OPS = OPT>=2, OPT>=2
PUSH_PERMUTES = OPT>=3  # TODO: reimplement this
UNSAFE_PAD_OPS = {BinaryOps.DIV, BinaryOps.CMPLT, UnaryOps.LOG2, UnaryOps.EXP2, UnaryOps.RECIP}

def _ast_reduceops(op:LazyOp) -> LazyOp:
  src = op.src[0]
  fuse = not src.realized and MERGE_ELEMENTWISE_INTO_REDUCE and isinstance(src, LazyBacking) and src.op.op in ElementwiseOps and len(src.children) <= 1
  return LazyOp(op.op, (src.op if fuse else src,), op.arg)

# this supports late merging an upstream Reduce op and even an Elementwise op above that
def _ast_binaryops(op:LazyOp) -> LazyOp:
  real_srcs: Dict[LazyCommon, Union[LazyOp, LazyCommon]] = {x:x for x in op.buffers}
  if MERGE_ONE_REDUCE_INTO_ELEMENTWISE:
    fuse_reduce = dedup([x for x in real_srcs.keys() if isinstance(x, LazyBacking) and not x.realized and x.op.op in ReduceOps and len(x.children) <= 1])
    if fuse_reduce:
      # NOTE: we only fuse the first reduce op if there's multiple fusable ones
      real_srcs[fuse_reduce[0]] = _ast_reduceops(fuse_reduce[0].op)
  return op.map_buffers(real_srcs)

"""
def fix_unbased_shape(src:LazyBuffer) -> LazyOp:
  assert src.is_contiguous(), "non contiguous can't be fixed"
  # if it's contiguous, applying a reshape to the base is safe
  return src.base.op.map_buffers({x:x.reshape(src.shape) for x in src.base.op.buffers}) if src.base != src else src.op
"""

"""
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
"""

def get_max_shape(x):
  if len(x) == 0: return None
  # TODO: this is wrong and they need to be combined
  return sorted(x, key=lambda x: len(x))[-1]

class LazyCommon:
  @property
  def buffers(self) -> Tuple[LazyBuffer, ...]: return (self,)
  def map_buffers(self, real_srcs: Mapping[LazyBuffer, Union[LazyBuffer, LazyOp]]): return real_srcs.get(self, self)
  def get_lazyops(self) -> List[LazyOp]: return []

# this is a contiguous array and can have an op fill it
class LazyBacking(LazyCommon):
  def __init__(self, op:Optional[LazyOp], size:int, dtype:DType, device:str, src:Optional[RawBuffer]=None):
    self.size, self.dtype, self.device = size, dtype, device
    self.realized: Optional[RawBuffer] = src
    self.output_buffer: Optional[RawBuffer] = None
    self.children: WeakSet = WeakSet()
    if op:
      self.op: LazyOp = op
      for x in op.buffers: x.children.add(self)

  def __repr__(self): return f"<LB {self.size} {self.dtype} op={(self.op.op if hasattr(self, 'op') else '') if not self.realized else self.realized}>"

  lazycache: WeakValueDictionary = WeakValueDictionary()
  @staticmethod
  def cache(op:LazyOp, size:int, dtype:DType, device:str):
    wop = (ref(op), size, dtype, device)
    if wop in LazyBacking.lazycache:
      for x in op.buffers: x.children.add(LazyBacking.lazycache[wop])
      return LazyBacking.lazycache[wop]
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
    return self.realize().realized.toCPU()
    """
    self_casted = self.e(UnaryOps.CAST, arg=(dtypes.from_np(self.dtype.np), False)) if dtypes.from_np(self.dtype.np) != self.dtype else self
    realized = self_casted.contiguous().realize().realized
    assert all_int(self.shape), f"no toCPU if shape is symbolic, {self.shape=}"
    return cast(RawBuffer, realized).toCPU()
    """

  def schedule(self, seen):
    if self in seen or self.realized: return []
    seen.add(self)

    op = self.op if self.op.op != LoadOps.CONTIGUOUS else LazyOp(UnaryOps.NOOP, self.op.src)
    if op.op in LoadOps: return [(self.op, self, ())]
    if op.op in ElementwiseOps: op = _ast_binaryops(op)
    elif op.op in ReduceOps: op = _ast_reduceops(op)

    base_bufs = dedup([x.base for x in op.buffers if x.realized or not isinstance(Device[x.device], Compiled) or x.device == "LLVM" or x.base.op.op != LoadOps.CONST])
    ret = []
    for x in base_bufs: ret += x.schedule(seen)

    reduceops: List[LazyOp] = dedup([x for x in op.get_lazyops() if x.op in ReduceOps])
    assert len(reduceops) <= 1, "max one reduce op in an ast"

    # get buffer shapes
    # TODO: remove_1s won't be needed once we have canonical LazyViews
    def remove_1s(x): return tuple([s for s in x if s != 1])
    if reduceops:
      late_shapes = [remove_1s(x.shape) for x in op.buffers if x not in reduceops[0].buffers]
      tmp_early_shape = get_common_shape([remove_1s(x.shape) for x in reduceops[0].buffers] + [reduceops[0].arg[0]])
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
      late_shape = get_common_shape([remove_1s(x.shape) for x in op.buffers])

    #all_shapes = [x.shape for x in op.buffers] + (list(reduceops[0].arg) if reduceops else [])
    #print(all_shapes)

    """
    early_shapes, late_shapes = [], []
    buffer_shapes = {}
    if reduceops:
      early_shapes.append(reduceops[0].arg[0])
      late_shapes.append(reduceops[0].arg[1])
      for x in reduceops[0].buffers: early_shapes.append(x.shape)
      early_shape = get_max_shape(early_shapes)
      for x in reduceops[0].buffers: buffer_shapes[x] = early_shape
    late_shapes += [x.shape for x in op.buffers if not reduceops or x not in reduceops[0].buffers]
    late_shape = get_max_shape(late_shapes)
    for x in op.buffers:
      if x not in buffer_shapes:
        buffer_shapes[x] = late_shape
    """

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

    return ret + [(op.map_buffers(replacements), self, base_bufs)]


    """
    if op.op in LoadOps: return [(self.op, self, ())]
    if op.op in ElementwiseOps: op = _ast_binaryops(op, self.shape)
    elif op.op in ReduceOps: op = _ast_reduceops(op)
    """
    #log_op(self, op.map_buffers({x:x.base for x in op.buffers}))

    #ret = []
    #for x in op.buffers: ret += x.schedule(seen)
    #return ret+[(_ast_bufferops(op), self, op.buffers)]

  def realize(self:LazyBuffer) -> LazyBuffer:
    if not self.realized:
      for op,out,buffers in self.schedule(set()):
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

  lazycache: WeakValueDictionary = WeakValueDictionary()
  @staticmethod
  def cache(st:ShapeTracker, base:LazyBacking):
    wop = (st, ref(base))
    if wop in LazyView.lazycache: return LazyView.lazycache[wop]
    LazyBacking.lazycache[wop] = ret = LazyView(st, base)
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
  def schedule(self, seen): return self.base.schedule(seen)
  def realize(self): return self.base.realize()
  def toCPU(self) -> np.ndarray: return LazyBuffer(self.shape, self).contiguous().backing.toCPU()
  @property
  def buffers(self) -> Tuple[LazyBuffer, ...]: return (self,)
  def map_buffers(self, real_srcs: Mapping[LazyBuffer, Union[LazyBuffer, LazyOp]]): return real_srcs.get(self, self)

# this is the normal LazyBuffer class. it has a shape and behaves like a reasonable tensor should
# there's no longer any point in caching it, as we just cache the underlying stuff
class LazyBuffer:
  def __init__(self, shape:Tuple[sint, ...], backing:Union[LazyBacking, LazyView]):
    assert isinstance(shape, tuple)
    self.shape, self.backing = shape, backing

  # passthrough properties
  @property
  def device(self): return self.backing.device
  @property
  def dtype(self): return self.backing.dtype
  @property
  def realized(self): return self.backing.realized
  def schedule(self): return self.backing.schedule(set())
  def realize(self): return self.backing.realize()

  """
  def __init__(self, op:Optional[LazyOp], st:ShapeTracker, dtype:DType, device:str, src:Optional[RawBuffer]=None, base:Optional[LazyBuffer]=None):
    self.st: ShapeTracker = st
    self.shape, self._dtype, self.device = self.st.shape, dtype, device
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
  """

  """
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
  """

  #def __repr__(self): return f"<L{'B' if self.base == self else 'V'} {self.shape} {self.dtype} op={(self.op.op if hasattr(self, 'op') else '') if not self.realized else self.realized} st={self.st}>"
  #def _device_extra_args(self) -> Dict[str, str]: return {"device": self.device.split(":", 1)[1]} if ":" in self.device else {}
  #def is_contiguous(self): return self.st.contiguous and self.base.st.size() == self.st.size()

  # handle base
  """
  @property
  def dtype(self): return self.base._dtype
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
  """

  def contiguous(self) -> LazyBuffer:
    if isinstance(self.backing, LazyBacking): return self
    return LazyBuffer(self.shape, LazyBacking(LazyOp(LoadOps.CONTIGUOUS, (self.backing,)), prod(self.shape), self.dtype, self.device))

  # *** external entrypoints ***

  @staticmethod
  def loadop(op, shape, dtype, device, arg=None, src:Optional[LazyBuffer]=None) -> LazyBuffer:
    return LazyBuffer(shape, LazyBacking(LazyOp(op, tuple() if src is None else (src.backing,), arg), prod(shape), dtype, device))
    #return LazyBuffer(LazyOp(op, tuple() if src is None else (src,), arg), ShapeTracker.from_shape(tuple(shape)), dtype, device)

  def const(self, val:Union[float, int]) -> LazyBuffer:
    return self.loadop(LoadOps.CONST, tuple(), dtypes.from_np(self.dtype.np), self.device, arg=val).reshape((1,)*len(self.shape)).expand(self.shape)

  @staticmethod
  def fromCPU(x: np.ndarray) -> LazyBuffer:
    return LazyBuffer(x.shape, LazyBacking(None, prod(x.shape), dtypes.from_np(x.dtype), "CPU", src=RawNumpyBuffer.fromCPU(x)))
    #return LazyBuffer(None, ShapeTracker.from_shape(x.shape), dtypes.from_np(x.dtype), "CPU", src=RawNumpyBuffer.fromCPU(x))

  def toCPU(self) -> np.ndarray: return self.backing.toCPU().reshape(self.shape)

  # *** elementwise ops ***

  def e(self:LazyBuffer, op:Union[UnaryOps, BinaryOps, TernaryOps], *srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
    bsrcs = tuple(x.backing for x in (self,)+srcs)
    out_dtype = max([x.dtype for x in bsrcs]) if op != UnaryOps.CAST else cast(Tuple[DType, bool], arg)[0]

    if MERGE_ELEMENTWISE_OPS:
      # remove the buffers from any (childless) BinaryOps that feed into this
      bsrcs = tuple([x.op if not x.realized and isinstance(x, LazyBacking) and x.op.op in ElementwiseOps and not x.children else x for x in bsrcs])  # type: ignore

    return LazyBuffer(self.shape, LazyBacking.cache(LazyOp(op, bsrcs, arg), prod(self.shape), out_dtype, self.device))

    """
    if MERGE_ELEMENTWISE_OPS:
      # remove the buffers from any (childless) BinaryOps that feed into this
      srcs = tuple([fix_unbased_shape(x) if not x.realized and x.is_contiguous() and x.base.op.op in ElementwiseOps and not x.children else x for x in srcs])  # type: ignore

    return LazyBuffer.cache(LazyOp(op, srcs, arg), ShapeTracker.from_shape(self.shape), out_dtype, self.device)
    """

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

  def _movement_op(self, fxn, arg, push_ewop=False, unsafe_ops=None) -> LazyBuffer:
    #print(self.shape, fxn, arg)
    st:ShapeTracker = fxn(self.backing.st.reshape(self.shape), arg)
    if st.contiguous and st.size() == self.backing.base.size: return LazyBuffer(st.shape, self.backing.base)
    if push_ewop and not self.realized and isinstance(self.backing, LazyBacking) and self.backing.op.op in ElementwiseOps:
      mapped = self.backing.op.map_buffers({x:LazyBuffer(self.shape, x)._movement_op(fxn, arg).backing for x in self.backing.op.buffers})
      return LazyBuffer(st.shape, LazyBacking.cache(mapped, prod(st.shape), self.backing.dtype, self.backing.device))
    return LazyBuffer(st.shape, LazyView.cache(st, self.backing.base))

    """
    st:ShapeTracker = fxn(self.st, arg)
    if self.st == st: return self
    if st == self.base.st: return self.base
    if push_ewop and not self.realized and self.is_contiguous() and self.base.op.op in ElementwiseOps:
      if not unsafe_ops or not any(x.op in unsafe_ops for x in self.base.op.get_lazyops()):
        if self.base == self:
          mapped = self.op.map_buffers({x:x._movement_op(fxn, arg) for x in self.op.buffers})
          return LazyBuffer.cache(mapped, ShapeTracker.from_shape(st.shape), self.dtype, self.device)
    return LazyBuffer.cache(None, st, self.dtype, self.device, base=self.base)
    """

  def permute(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.permute, arg, SHUFFLE_MOVEMENT_OPS)
  def shrink(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.shrink, arg, SHUFFLE_MOVEMENT_OPS)
  def stride(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.stride, arg, SHUFFLE_MOVEMENT_OPS)
  def pad(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.pad, arg) #, SHUFFLE_PAD_OPS, UNSAFE_PAD_OPS)
  def reshape(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.reshape, arg) #, SHUFFLE_MOVEMENT_OPS)
  def expand(self, arg) -> LazyBuffer: return self._movement_op(ShapeTracker.expand, arg)

  def map_buffers(self, real_srcs: Mapping[LazyBuffer, Union[LazyBuffer, LazyOp]]): return real_srcs.get(self, self)
  def get_lazyops(self) -> List[LazyOp]: return []

  """
  def schedule(self:LazyBuffer, seen=None) -> List[Tuple[LazyOp, LazyBuffer, Tuple[LazyBuffer, ...]]]:
    if seen is None: seen = set()
    if self in seen: return []
    seen.add(self)
    if self.realized: return []
    #if self.base != self: return self.base.schedule(seen)

    # NOTE: late rewrite contiguous
    op = self.op if self.op.op != LoadOps.CONTIGUOUS else LazyOp(UnaryOps.NOOP, self.op.src)

    if op.op in LoadOps: return [(self.op, self, ())]
    if op.op in ElementwiseOps: op = _ast_binaryops(op, self.shape)
    elif op.op in ReduceOps: op = _ast_reduceops(op)
    ret = []
    for x in op.buffers: ret += x.schedule(seen)
    log_op(self, op.map_buffers({x:x.base for x in op.buffers}))
    return ret+[(_ast_bufferops(op), self, op.buffers)]

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
          out.realized = Device[buffers[0].device].exec_ast(op, output=out, inputs=realized_bufs, var_vals={}, **self._device_extra_args())
          del out.op
        assert out.realized.dtype == out.dtype, "realized dtype is correct"
    return self
  """

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