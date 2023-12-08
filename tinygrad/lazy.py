from __future__ import annotations
import sys, math
from typing import Callable, Optional, Tuple, Union, List, Dict, Any, cast, Mapping, Set
from weakref import ref, WeakSet, WeakValueDictionary

import numpy as np
from tinygrad.helpers import prod, getenv, DType, dtypes, flatten, dedup, merge_dicts, all_int, ImageDType, DEBUG
from tinygrad.ops import ScheduleItem, UnaryOps, BinaryOps, TernaryOps, ReduceOps, MovementOps, LoadOps, OpType, LazyOp, MemBuffer, ConstBuffer, BufferOps, get_lazyop_info, vars_from_ast
from tinygrad.shape.shapetracker import ShapeTracker, get_contraction
from tinygrad.shape.symbolic import sint
from tinygrad.device import Buffer

# lazy can recurse a lot
sys.setrecursionlimit(10000)

OPT = getenv("OPT", 2)
LAZYCACHE = getenv("LAZYCACHE", 1)

# TODO: movement ops that only change shape are really nops. treat them as such
REMOVE_MOVEMENT_NOPS, MERGE_ELEMENTWISE_INTO_REDUCE, SHUFFLE_MOVEMENT_OPS, MERGE_ELEMENTWISE_OPS = OPT>=1, OPT>=1, OPT>=1, OPT>=1
MERGE_ONE_REDUCE_INTO_ELEMENTWISE, SHUFFLE_PAD_OPS = OPT>=2, OPT>=2
PUSH_PERMUTES, PUSH_CONTIGUOUS = OPT>=3, OPT>=3
PUSH_RESHAPES = OPT>=4

# **** ast fixing functions ****

def _ast_reduceops(op:LazyOp) -> LazyOp:
  # TODO: this can also corealize a binary op after the reduce, not just before
  src = op.src[0]
  if not src.realized:
    assert isinstance(src.op, LazyOp), "if not src.realized, then src.op must be a LazyOp"
    if MERGE_ELEMENTWISE_INTO_REDUCE and src.optype is BinaryOps and len(src.children) <= 1: src = src.op
  return LazyOp(op.op, (src,), op.arg)

# this supports late merging an upstream Reduce op and even an Elementwise op above that
def _ast_binaryops(op:LazyOp, shape:Tuple[sint, ...]) -> LazyOp:
  real_srcs: Dict[LazyBuffer, Optional[Union[LazyOp, LazyBuffer]]] = {x:None for x in op.buffers}
  # NOTE: contiguous does not always mean the same size with SHRINK. this is still mergeable but requires more thought how
  # TODO: this can also support late fusion of BinaryOps, required for test_fold_conv_sgd
  psrcs = [(buf,root) for buf in op.buffers if len(buf.children) <= 1 and (root:=get_movementroot_contiguous(buf)).optype == ReduceOps and not root.realized and prod(root.shape) == prod(buf.shape) and len(root.children) <= 1]
  intermediate_shape = shape
  if MERGE_ONE_REDUCE_INTO_ELEMENTWISE and psrcs:
    # NOTE: right now we can't handle multiple, as we'd have to check for loop
    buf,root = psrcs[0]
    top = _ast_reduceops(root.op)
    real_srcs[buf] = top
    real_srcs.update({x:x for x in top.buffers})  # the reduce op buffers are not modified

    # if the ReduceOp is followed by a reshape, we push this reshape before all the ElementwiseOp inputs
    if buf.shape != root.shape:
      intermediate_shape = root.shape
      assert buf.shape == shape, f"shape mismatch {buf.shape} != {shape}"

  # reshape all the late ops into the output shape
  # NOTE: these RESHAPEs will return self if they don't change the shape
  for buf,src in real_srcs.items():
    if src is None: real_srcs[buf] = buf.reshape(intermediate_shape)
  # NOTE: cast the type to remove the Optional
  ast = op.map_buffers(cast(Dict[LazyBuffer, Union[LazyOp, LazyBuffer]], real_srcs))
  return LazyOp(MovementOps.RESHAPE, (ast, ), shape) if intermediate_shape != shape else ast

def _replace_bufferops(op:LazyOp) -> Tuple[LazyOp, List[LazyBuffer]]:
  replacements:Dict[LazyBuffer, LazyOp] = {}
  base_bufs = dedup([x.base for x in op.buffers if not x.is_unrealized_const()])
  for x in op.buffers:
    st = x.st.simplify().unbind()
    if x.base in base_bufs:
      replacements[x] = LazyOp(BufferOps.LOAD, (), MemBuffer(base_bufs.index(x.base)+1, x.dtype, st))
    elif not x.realized and x.base.op.op == LoadOps.CONST:
      replacements[x] = LazyOp(BufferOps.CONST, (), ConstBuffer(float(x.base.op.arg), x.dtype, st))
    else:
      raise NotImplementedError(f"not handled {x}")
  return (op.src[0] if op.op in {MovementOps.RESHAPE, LoadOps.CONTIGUOUS} else op).map_buffers(replacements), base_bufs

# **** lazy operations ****

def get_movementroot(root:LazyBuffer, allow_contiguous=False) -> LazyBuffer: return get_movementroot(cast(LazyBuffer, root.op.src[0]), allow_contiguous) if not root.realized and (root.optype == MovementOps or (root.op.op == LoadOps.CONTIGUOUS and allow_contiguous and root.op.src[0].st.contiguous)) else root
def get_movementroot_contiguous(x:LazyBuffer) -> LazyBuffer: return get_movementroot_contiguous(cast(LazyBuffer, x.op.src[0])) if not x.realized and x.op.op == LoadOps.CONTIGUOUS else (get_movementroot(x, True) if x.optype == MovementOps and x.st.contiguous else x)

# NOTE: this is the canonical order


lazycache: WeakValueDictionary = WeakValueDictionary()
def create_lazybuffer(device:str, st:ShapeTracker, optype:OpType, op:LazyOp, dtype:DType, base:Optional[LazyBuffer]=None):
  # rewrite 0 size into a CONST
  if 0 in st.shape: return LazyBuffer(device, ShapeTracker.from_shape(st.shape), LoadOps, LazyOp(LoadOps.CONST, tuple(), 0.0), dtype)

  # fromcpu aren't cached
  if not LAZYCACHE or (optype is LoadOps and op.op in {LoadOps.EMPTY, LoadOps.CUSTOM, LoadOps.CONST}): return LazyBuffer(device, st, optype, op, dtype, base=base)

  # wop is the deduping key. i feel this used to compare more deeply
  wop = (device, dtype, optype, ref(op), ref(base) if base else None)
  if wop in lazycache:
    for x in op.buffers: x.children.add(lazycache[wop])
    return lazycache[wop]

  lazycache[wop] = ret = LazyBuffer(device, st, optype, op, dtype, base=base)
  return ret

class LazyBuffer:
  __deletable__ = ('op',)
  def __init__(self, device:str, st:ShapeTracker, optype:OpType, op:Optional[LazyOp], dtype:DType, src:Optional[Buffer]=None, base:Optional[LazyBuffer]=None):
    self.device, self.st, self.shape, self.optype, self._dtype, self._realized = device, st, st.shape, optype, dtype, src
    self.output_buffer: Optional[Buffer] = None   # TODO: do we really need this? or can we just use realized
    # TODO: does children have to be a ref count instead of a set? can a Buffer be a double child?
    self.children: WeakSet[LazyBuffer] = WeakSet()
    self.views: WeakSet[LazyBuffer] = WeakSet()
    # NOTE: op should be read only after construction of LazyBuffer. it is now with schedule
    if op is not None:
      self.op = op
      for x in op.buffers: x.children.add(self)
    assert optype != MovementOps or (base is not None and base.optype != MovementOps), "MovementOps must be based"
    self._base = base
    if base: base.views.add(self)
    else: assert st.contiguous, "unbased LazyBuffers must be contiguous"

  @property
  def base(self): return self._base if self._base is not None else self

  def is_unrealized_const(self): return not self.realized and self.base.op.op == LoadOps.CONST
  def is_unrealized_contiguous_const(self): return self.is_unrealized_const() and self.st.contiguous

  @property
  def realized(self): return self.base._realized
  @realized.setter
  def realized(self, val:Buffer):
    assert self._base is None, "no setting realized of based LazyBuffers"
    self._realized = val
  @property
  def dtype(self): return self.base._dtype
  @dtype.setter
  def dtype(self, val:DType):
    assert self._base is None, "no setting dtype of based LazyBuffers"
    self._dtype = val

  def __repr__(self): return f"<LB {self.shape} {self.dtype} op={self.op.op if hasattr(self, 'op') else self._realized} st={self.st}>"

  def _device_extra_args(self) -> Dict[str, str]: return {"device": self.device.split(":", 1)[1]} if ":" in self.device else {}

  @property
  def buffers(self) -> Tuple[LazyBuffer, ...]: return (self,)
  def map_buffers(self, real_srcs: Mapping[Any, Union[LazyBuffer, LazyOp]]): return real_srcs.get(self, self)
  def get_lazyops(self) -> List[LazyOp]: return []

  # *** scheduling ***

  def schedule(self, seen:Optional[Set[LazyBuffer]]=None) -> List[ScheduleItem]:
    if seen is None: seen = set()
    if self in seen or self.realized or self.is_unrealized_const(): return []
    seen.add(self)
    if self.base is not self: return self.base.schedule(seen)

    op = self.op
    if self.optype is BinaryOps: op = _ast_binaryops(op, self.shape)
    elif self.optype is ReduceOps: op = _ast_reduceops(op)

    # schedule the past
    ret:List[ScheduleItem] = []
    for x in op.buffers: ret += x.schedule(seen)

    var_vals = merge_dicts([self.st.var_vals] + [buf.st.var_vals for buf in op.buffers])

    op, base_bufs = _replace_bufferops(op)

    # check if we can reuse the output buffer
    # if it's aliased, don't use it
    # TODO: this is pretty wrong actually, who knows where else this buffer is used?
    # TODO: what if an assign is required? this silently is wrong
    # NOTE: this has been moved to schedule, as this is only an issue if buffers are already realized
    if self.output_buffer is not None:
      for i,a in enumerate(base_bufs):
        # TODO: if this is contiguous it's fine
        if a.realized == self.output_buffer:
          if any(not x.arg.st.contiguous for x in op.get_lazyops() if x.op == BufferOps.LOAD and x.arg.idx == i+1):
            self.output_buffer = None
            break

    if op.op not in LoadOps:
      # add the store
      info = get_lazyop_info(op)
      assert info.dtype == self.dtype or isinstance(self.dtype, ImageDType), f"dtype mismatch {info.dtype=} != {self.dtype=}"

      if isinstance(self.dtype, ImageDType) and (prod(self.shape) != prod(self.dtype.shape) or not any(self.shape[x]%4 == 0 for x in self.st.unit_stride_axes())):
        if DEBUG >= 3: print(f"forcing image {self.dtype} to float32")
        self.dtype = dtypes.float32  # NOTE; this is what makes the dtype above not match
        op = LazyOp(UnaryOps.CAST, (op, ), (dtypes.float32, False))

      # TODO: why doesn't this match?
      #assert info.shape == self.shape, f"shape mismatch {info.shape=} != {self.shape=}"
      op = LazyOp(BufferOps.STORE, (op, ), MemBuffer(0, self.dtype, ShapeTracker.from_shape(info.shape)))
    else:
      # check loadop validity of bufferops
      for i,s in enumerate(op.src): assert isinstance(s, LazyOp) and s.op == BufferOps.LOAD and s.arg.idx == i+1 and s.arg.st.contiguous, f"bad LoadOps src {i}: {s}"

    return ret + [ScheduleItem(op, self, tuple(base_bufs), {k:var_vals[k] for k in vars_from_ast(op)})]

  # *** creation/special ops ***

  @staticmethod
  def loadop(op, shape:Tuple[sint,...], dtype:DType, device:str, arg=None, src:Optional[LazyBuffer]=None) -> LazyBuffer:
    return create_lazybuffer(device, ShapeTracker.from_shape(shape), LoadOps, LazyOp(op, tuple() if src is None else (src,), arg), dtype)

  # create a constant with the shape and dtype of self
  def const(self, val:Union[float, int]) -> LazyBuffer:
    # NOTE: dtypes.from_np(self.dtype.np) to deal with image types
    return LazyBuffer.loadop(LoadOps.CONST, tuple(), dtypes.from_np(self.dtype.np), self.device, arg=val).reshape((1,)*len(self.shape)).expand(self.shape)

  def copy_to_device(self, device:str) -> LazyBuffer:
    # back off a COPY if it's a double COPY
    if not self.realized and self.op.op == LoadOps.COPY and cast(LazyBuffer, self.op.src[0]).device == device: return cast(LazyBuffer, self.op.src[0])
    return LazyBuffer.loadop(LoadOps.COPY, self.shape, self.dtype, device, src=self.contiguous())

  def contiguous(self:LazyBuffer) -> LazyBuffer:
    if not self.realized and self.op.op in LoadOps and self.op.op != LoadOps.CONST: return self  # all LoadOps are already contiguous (except CONST)
    if self.st.contiguous and self.st.size() == self.base.st.size() and not self.is_unrealized_const():
      # this will turn into nothing, it's based and a copy
      # TODO: based lazybuffers shouldn't take dtype or var_vals, same issue in movementops
      return create_lazybuffer(self.device, ShapeTracker.from_shape(tuple(self.shape)), LoadOps, LazyOp(LoadOps.CONTIGUOUS, (self,), None), self.dtype, base=self.base)
    return LazyBuffer.loadop(LoadOps.CONTIGUOUS, self.shape, self.dtype, self.device, src=self)

  @staticmethod
  def fromCPU(x: np.ndarray) -> LazyBuffer:
    return LazyBuffer("CPU", ShapeTracker.from_shape(x.shape), LoadOps, None, dtypes.from_np(x.dtype), Buffer("CPU", prod(x.shape), dtypes.from_np(x.dtype), x.flatten()))

  def cast(self, dtype:DType, bitcast:bool=False):
    return self.e(UnaryOps.CAST, arg=(dtype, bitcast))

  # *** elementwise ops ***

  def e(self:LazyBuffer, op:Union[UnaryOps, BinaryOps, TernaryOps], *srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
    # srcs includes self
    srcs = (self,)+srcs

    # if we are separated from other binary ops by movement ops, we push those movement ops above those binaryops
    if SHUFFLE_MOVEMENT_OPS: srcs = _push_movement_ops(srcs)

    # get outputs now
    out_device, out_shape, out_dtype = srcs[0].device, srcs[0].shape, max([x.dtype for x in srcs]) if op != UnaryOps.CAST else cast(Tuple[DType, bool], arg)[0]

    # push all contiguous to the end of BinaryOps
    if PUSH_CONTIGUOUS and any(not x.realized and x.op.op == LoadOps.CONTIGUOUS and len(x.op.src[0].children) <= 1 for x in srcs):
      new_srcs: List[LazyBuffer] = []
      for x in srcs:
        if not x.realized and x.op.op == LoadOps.CONTIGUOUS and len(x.op.src[0].children) <= 1:
          x.op.src[0].children.discard(x)
          x = cast(LazyBuffer, x.op.src[0])
        new_srcs.append(x)
      return new_srcs[0].e(op, *new_srcs[1:], arg=arg).contiguous()

    if MERGE_ELEMENTWISE_OPS:
      # remove the buffers from any (childless) BinaryOps that feed into this
      _srcs = tuple([x.op if x.optype == BinaryOps and not x.children and not x.realized else x for x in srcs])
      # TODO: needs general merge limiting
      if out_device != "WEBGPU" or len(dedup([x.base for _src in _srcs for x in _src.buffers if not x.is_unrealized_const()])) < 7: srcs = _srcs # type: ignore

    return create_lazybuffer(out_device, ShapeTracker.from_shape(out_shape), BinaryOps, LazyOp(op, srcs, arg), out_dtype)

  # *** reduce ops ***

  def _reduce_op(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[sint, ...]) -> LazyBuffer:
    if self.shape == tuple(new_shape): return self
    srcs = _push_movement_ops((self,)) if SHUFFLE_MOVEMENT_OPS else (self,)
    unbound_new_shape = tuple(s.unbind()[0] if not isinstance(s, int) else s for s in new_shape)
    return create_lazybuffer(self.device, ShapeTracker.from_shape(new_shape), ReduceOps, LazyOp(op, srcs, unbound_new_shape), self.dtype)

  def r(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[sint, ...]) -> LazyBuffer:
    # TODO: can we split symbolic shape if the reduce axis is not symbolic?
    if not all_int(self.shape) or (0 in self.shape) or prod(self.shape) // prod(new_shape) < getenv("REDUCEOP_SPLIT_THRESHOLD", 32768): return self._reduce_op(op, new_shape)
    heuristic, divisor, dim_to_split = max(((divisor := math.gcd(256, old))/(stride or math.inf), divisor, i) for i, (old, new, stride) in enumerate(zip(self.shape, new_shape, self.st.real_strides())) if old != new) # type: ignore
    if divisor < 16 or heuristic < 0.1: return self._reduce_op(op, new_shape)
    # choose largest divisor (>=16) to split on, penalize large strides
    def splitted_shape(dim_aft_div): return self.shape[:dim_to_split] + (self.shape[dim_to_split]//divisor,) + dim_aft_div + self.shape[dim_to_split+1:]
    return self.reshape(splitted_shape((divisor,)))._reduce_op(op, splitted_shape((1,))).reshape(splitted_shape(()))._reduce_op(op, new_shape)

  # *** movement ops ***

  def reshape(self:LazyBuffer, arg:Tuple[sint, ...]) -> LazyBuffer:
    if self.shape == arg: return self
    if not self.realized and self.op.op == MovementOps.RESHAPE:
      assert isinstance(self.op.src[0], LazyBuffer)
      self.op.src[0].children.discard(self) # NOTE: this is only required in reshape and when pushing permutes, why??
      return self.op.src[0].reshape(arg)
    return self._movement_op(self.st.reshape(arg), MovementOps.RESHAPE, arg)

  def pad(self:LazyBuffer, arg:Tuple[Tuple[int, int], ...]) -> LazyBuffer:
    if all(b == 0 and e == 0 for b,e in arg): return self
    if not self.realized and self.op.op == MovementOps.PAD: return self.op.src[0].pad(tuple([(b1+b2, e1+e2) for (b1,e1),(b2,e2) in zip(self.op.arg, arg)]))
    return self._movement_op(self.st.pad(arg), MovementOps.PAD, arg)

  def expand(self: LazyBuffer, arg:Tuple[sint, ...]) -> LazyBuffer:
    if self.shape == arg: return self
    if not self.realized and self.op.op == MovementOps.EXPAND: return self.op.src[0].expand(arg)
    return self._movement_op(self.st.expand(arg), MovementOps.EXPAND, arg)

  def permute(self: LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer:
    if arg == tuple(range(len(self.shape))): return self
    if not self.realized and self.op.op == MovementOps.PERMUTE: return self.op.src[0].permute(tuple([self.op.arg[i] for i in arg]))
    if SHUFFLE_MOVEMENT_OPS and not self.realized:
      if PUSH_PERMUTES and self.optype == ReduceOps:
        # reduceops have one buffer input, permute it
        narg = tuple([self.op.arg[a] for a in arg])
        src, rop = self.op.src[0], self.op.op
        src.children.discard(self)
        del self  # TODO: why doesn't this delete remove it from the children
        return src.permute(arg).r(cast(ReduceOps, rop), narg)

      # move permutes before expands (always, this is safe)
      if self.op.op == MovementOps.EXPAND:
        return self.op.src[0].permute(arg).expand(tuple([self.op.arg[a] for a in arg]))

      # move permutes before reshapes if we can
      if PUSH_PERMUTES and self.op.op == MovementOps.RESHAPE and isinstance(self.op.src[0], LazyBuffer):
        if shape_idx_groups := get_contraction(self.op.src[0].shape, self.shape):
          self.op.src[0].children.discard(self) # NOTE: this is only required in reshape and when pushing permutes, why??
          return self.op.src[0].permute(tuple(flatten(shape_idx_groups[i] for i in arg))).reshape(self.st.permute(arg).shape)
    return self._movement_op(self.st.permute(arg), MovementOps.PERMUTE, arg)

  def shrink(self:LazyBuffer, arg:Tuple[Tuple[sint, sint], ...]) -> LazyBuffer:
    if all(b - a == s for s, (a, b) in zip(self.shape, arg)): return self
    if not self.realized and self.op.op == MovementOps.SHRINK: return self.op.src[0].shrink(tuple([(b1+b2, b1+e2) for (b1,_),(b2,e2) in zip(self.op.arg, arg)]))
    return self._movement_op(self.st.shrink(arg), MovementOps.SHRINK, arg)

  def stride(self:LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer:
    if all(a == 1 for a in arg): return self
    if not self.realized and self.op.op == MovementOps.STRIDE: return self.op.src[0].stride(tuple(a1*a2 for a1,a2 in zip(arg, self.op.arg)))
    return self._movement_op(self.st.stride(arg), MovementOps.STRIDE, arg)

  def _movement_op(self, st: ShapeTracker, op: MovementOps, arg: Union[Tuple[sint, ...], Tuple[Tuple[sint, sint], ...]]) -> LazyBuffer:
    if SHUFFLE_MOVEMENT_OPS and not self.realized and self.optype == BinaryOps and not self.children:
      if op in {MovementOps.SHRINK, MovementOps.STRIDE, MovementOps.PERMUTE} or (op == MovementOps.RESHAPE and (self.op.op in UnaryOps or PUSH_RESHAPES)):
        return self.op.replace_with_movement_ops([(op, arg)])
    if REMOVE_MOVEMENT_NOPS and not self.realized and st.contiguous:
      # MovementOps aren't stacked any more, they each have one parent, find the root
      if (root:=get_movementroot(self)) != self and root.st.contiguous and prod(st.shape) == prod(root.shape):
        return root.reshape(st.shape)
    return create_lazybuffer(self.device, st, MovementOps, LazyOp(op, (self,), arg), self.dtype, base=self.base)

  def replace_with_movement_ops(self: LazyBuffer, ops:List[Tuple[MovementOps, Any]]) -> LazyBuffer:
    y = self
    for op, arg in ops: y = MOVEMENT_OPS_DISPATCHER[op](y, arg)
    return y

UNSAFE_PAD_OPS = {BinaryOps.DIV, BinaryOps.CMPLT, UnaryOps.LOG2, UnaryOps.EXP2, UnaryOps.RECIP}

def _push_movement_ops(srcs:Tuple[LazyBuffer, ...]) -> Tuple[LazyBuffer, ...]:
  new_srcs = []
  for x in srcs:
    mops: List[Tuple[MovementOps, Any]] = []
    bx = x
    # backwalk all the movement ops. don't push PAD or EXPAND
    while not bx.realized and bx.optype is MovementOps and bx.op.op is not MovementOps.EXPAND and (SHUFFLE_PAD_OPS or bx.op.op is not MovementOps.PAD) and len(bx.children) <= 1:
      assert isinstance(bx.op.op, MovementOps) and isinstance(bx.op.src[0], LazyBuffer)
      mops.append((bx.op.op, bx.op.arg))
      bx = bx.op.src[0]
    # NOTE: can't push pads past anything where f(0, 0) != 0 or f(0) != 0
    if mops and not bx.realized and bx.optype is BinaryOps and len(bx.children) <= 1 and (all(y[0] is not MovementOps.PAD for y in mops) or all(y.op not in UNSAFE_PAD_OPS for y in bx.op.get_lazyops())):
      x = bx.op.replace_with_movement_ops(mops[::-1])
    new_srcs.append(x)
  return tuple(new_srcs)

MOVEMENT_OPS_DISPATCHER: Dict[MovementOps, Callable] = {
  MovementOps.RESHAPE: LazyBuffer.reshape, MovementOps.EXPAND: LazyBuffer.expand, MovementOps.SHRINK: LazyBuffer.shrink,
  MovementOps.PERMUTE: LazyBuffer.permute, MovementOps.PAD: LazyBuffer.pad, MovementOps.STRIDE: LazyBuffer.stride,
}
