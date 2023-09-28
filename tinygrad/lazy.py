from __future__ import annotations
import sys, operator, math
from typing import Callable, Optional, Tuple, Union, List, Dict, Any, cast, Mapping
from weakref import ref, WeakSet, WeakValueDictionary

import numpy as np
from tinygrad.graph import log_op
from tinygrad.helpers import GRAPH, DEBUG, prod, getenv, DType, dtypes, flatten, ImageDType, partition, all_int, dedup, merge_dicts
from tinygrad.ops import Device, Compiled, UnaryOps, BinaryOps, TernaryOps, ReduceOps, MovementOps, LoadOps, OpType, LazyOp, MemBuffer, ConstBuffer, BufferOps
from tinygrad.shape.shapetracker import ShapeTracker, get_contraction
from tinygrad.shape.symbolic import Variable, sint

from tinygrad.runtime.lib import RawConst, RawBuffer, RawBufferMapped, RawBufferTransfer
from tinygrad.runtime.ops_cpu import RawNumpyBuffer
from tinygrad.runtime.ops_disk import RawDiskBuffer

# lazy can recurse a lot
sys.setrecursionlimit(10000)

OPT = getenv("OPT", 2)
LAZY = getenv("LAZY", 1)
LAZYCACHE = getenv("LAZYCACHE", 1)
P2P = getenv("P2P", 0)

# TODO: movement ops that only change shape are really nops. treat them as such
REMOVE_MOVEMENT_NOPS, MERGE_ELEMENTWISE_INTO_REDUCE, SHUFFLE_MOVEMENT_OPS, MERGE_ELEMENTWISE_OPS = OPT>=1, OPT>=1, OPT>=1, OPT>=1
MERGE_ONE_REDUCE_INTO_ELEMENTWISE, SHUFFLE_PAD_OPS = OPT>=2, OPT>=2
PUSH_PERMUTES, PUSH_CONTIGUOUS = OPT>=3, OPT>=3

# **** realize functions ****
def _ast_reduceops(op:LazyOp) -> LazyOp:
  # TODO: this can also corealize a binary op after the reduce, not just before
  src = op.src[0]
  if not src.realized:
    assert isinstance(src.op, LazyOp), "if not src.realized, then src.op must be a LazyOp"
    if MERGE_ELEMENTWISE_INTO_REDUCE and src.optype is BinaryOps and len(src.children) <= 1: src = src.op
  return LazyOp(op.op, (src,), op.arg)

# this supports late merging an upstream Reduce op and even an Elementwise op above that
def _ast_binaryops(op:LazyOp, shape: Tuple[sint, ...]) -> LazyOp:
  real_srcs: Dict[LazyBuffer, Optional[Union[LazyOp, LazyBuffer]]] = {x:None for x in op.buffers}
  # NOTE: contiguous does not always mean the same size with SHRINK. this is still mergeable but requires more thought how
  # TODO: this can also support late fusion of BinaryOps, required for test_fold_conv_sgd
  psrcs: List[Tuple[LazyBuffer, LazyBuffer]] = [(k,x) for k,x in zip(real_srcs.keys(), map(get_movementroot_contiguous, real_srcs.keys())) if x.optype == ReduceOps and not x.realized and prod(k.shape) == prod(x.shape) and len(x.children) <= 1 and len(k.children) <= 1]
  intermediate_shape: Tuple[sint, ...] = shape
  if MERGE_ONE_REDUCE_INTO_ELEMENTWISE and psrcs:
    psrc = psrcs[0] # NOTE: right now we can't handle multiple, as we'd have to check for loop
    if psrc[1].optype == ReduceOps:
      top = _ast_reduceops(psrc[1].op)
    real_srcs[psrc[0]] = top
    real_srcs.update({x:x for x in top.buffers})  # the reduce op buffers are not modified

    # if the ReduceOp is followed by a reshape, we push this reshape before all the ElementwiseOp inputs
    if psrc[0].shape != psrc[1].shape:
      intermediate_shape = psrc[1].shape
      assert psrc[0].shape == shape, f"shape mismatch {psrc[0].shape} != {shape}"

  # reshape all the late ops into the output shape
  # NOTE: these RESHAPEs will return self if they don't change the shape
  for x in real_srcs.keys():
    if real_srcs[x] is None: real_srcs[x] = x.reshape(intermediate_shape)
  # NOTE: cast the type to remove the Optional
  ast = op.map_buffers(cast(Dict[LazyBuffer, Union[LazyOp, LazyBuffer]], real_srcs))
  return LazyOp(MovementOps.RESHAPE, (ast, ), shape) if intermediate_shape != shape else ast

def _replace_bufferops(op:LazyOp) -> Tuple[LazyOp, List[LazyBuffer]]:
  replacements:Dict[LazyBuffer, LazyOp] = {}
  base_bufs = dedup([x.base for x in op.buffers if (x.realized and not isinstance(x.realized, RawConst)) or not isinstance(Device[x.device], Compiled) or x.device == "LLVM" or x.base.op.op != LoadOps.CONST])
  for x in op.buffers:
    st = x.st.simplify()
    if x.base in base_bufs:
      replacements[x] = LazyOp(BufferOps.MEM, (), MemBuffer(base_bufs.index(x.base)+1, x.dtype, st))
    elif x.realized and isinstance(x.realized, RawConst):
      replacements[x] = LazyOp(BufferOps.CONST, (), ConstBuffer(x.realized._buf, x.realized.dtype, st))
    elif not x.realized and x.base.op.op == LoadOps.CONST:
      replacements[x] = LazyOp(BufferOps.CONST, (), ConstBuffer(float(x.base.op.arg), x.dtype, st))
    else:
      raise NotImplementedError(f"not handled {x}")
  return (op.src[0] if op.op == MovementOps.RESHAPE else op).map_buffers(replacements), base_bufs

# **** lazy operations ****

def get_single_root(root:LazyBuffer) -> LazyBuffer: return get_single_root(cast(LazyBuffer, root.op.src[0])) if getattr(root, 'op', None) and len(root.op.src) == 1 else root
def get_movementroot(root:LazyBuffer, allow_contiguous=False) -> LazyBuffer: return get_movementroot(cast(LazyBuffer, root.op.src[0]), allow_contiguous) if not root.realized and (root.optype == MovementOps or (root.op.op == LoadOps.CONTIGUOUS and allow_contiguous and root.op.src[0].st.contiguous)) else root
def get_movementroot_contiguous(x:LazyBuffer) -> LazyBuffer: return get_movementroot_contiguous(cast(LazyBuffer, x.op.src[0])) if not x.realized and x.op.op == LoadOps.CONTIGUOUS else (get_movementroot(x, True) if x.optype == MovementOps and x.st.contiguous else x)

lazycache: WeakValueDictionary = WeakValueDictionary()
def create_lazybuffer(device:str, st:ShapeTracker, optype:OpType, op:LazyOp, dtype:DType, var_vals:Dict[Variable,int], base:Optional[LazyBuffer]=None):
  # fromcpu aren't cached
  if not LAZYCACHE or (optype is LoadOps and op.op in {LoadOps.EMPTY, LoadOps.RAND, LoadOps.CONST}): return LazyBuffer(device, st, optype, op, dtype, var_vals, base=base)

  # wop is the deduping key. i feel this used to compare more deeply
  wop = (device, dtype, optype, ref(op), tuple(sorted(var_vals.keys())), ref(base) if base else None)
  if wop in lazycache:
    for x in op.buffers: x.children.add(lazycache[wop])
    return lazycache[wop]

  lazycache[wop] = ret = LazyBuffer(device, st, optype, op, dtype, var_vals, base=base)
  return ret

UNSAFE_PAD_OPS = {BinaryOps.DIV, BinaryOps.CMPLT, UnaryOps.LOG2, UnaryOps.EXP2, UnaryOps.RECIP}

class LazyBuffer:
  __deletable__ = ('op',)
  def __init__(self, device:str, st:ShapeTracker, optype:OpType, op:LazyOp, dtype:DType, var_vals:Dict[Variable,int], src:Optional[RawBuffer]=None, base:Optional[LazyBuffer]=None):
    self.st: ShapeTracker = st  # NOTE: this is not a copy! this should be a "read-only" ShapeTracker
    self._var_vals: Dict[Variable, int] = var_vals
    self.device, self.shape, self.optype, self._dtype = device, self.st.shape, optype, dtype
    self._realized: Optional[RawBuffer] = src
    self.output_buffer: Optional[RawBuffer] = None   # TODO: do we really need this? or can we just use realized
    # TODO: does children have to be a ref count instead of a set? can a Buffer be a double child?
    self.children: WeakSet = WeakSet()
    # NOTE: op should be read only after construction of LazyBuffer
    self.op: LazyOp = op
    assert optype != MovementOps or (base is not None and base.optype != MovementOps), "MovementOps must be based"
    self._base = base
    for x in op.buffers: x.children.add(self)
    if not LAZY: self.realize()

    # log phantom ops to the graph
    if GRAPH >= 3:
      log_op(self, self.op, phantom=True)

  @property
  def var_vals_key(self): return tuple(sorted(self.var_vals.keys()))

  @property
  def base(self): return self._base if self._base is not None else self

  @property
  def realized(self): return self.base._realized
  @realized.setter
  def realized(self, val):
    assert self._base is None, "no setting realized of based LazyBuffers"
    self._realized = val
  @property
  def dtype(self): return self.base._dtype
  @dtype.setter
  def dtype(self, val):
    assert self._base is None, "no setting dtype of based LazyBuffers"
    self._dtype = val
  @property
  def var_vals(self): return self.base._var_vals
  @var_vals.setter
  def var_vals(self, val):
    assert self._base is None, "no setting var_vals of based LazyBuffers"
    self._var_vals = val

  def __repr__(self): return f"<LB {self.shape} {self.dtype} op={self.op.op if not self._realized else self._realized} st={self.st}>"
  @property
  def key(self):
    if self.realized: return (self.dtype, self.realized.key, self.st, self.var_vals_key)
    return (self.dtype, self.op.op, self.st, self.var_vals_key)

  def _device_extra_args(self) -> Dict[str, str]: return {"device": self.device.split(":", 1)[1]} if ":" in self.device else {}

  @property
  def buffers(self) -> Tuple[LazyBuffer, ...]: return (self,)
  def map_buffers(self, real_srcs: Mapping[LazyBuffer, Union[LazyBuffer, LazyOp]]): return real_srcs.get(self, self)
  def get_lazyops(self) -> List[LazyOp]: return []

  def schedule(self, seen=None):
    if seen is None: seen = set()
    if self in seen or self.realized: return []
    seen.add(self)

    op = self.op if self.op.op != LoadOps.CONTIGUOUS else LazyOp(UnaryOps.NOOP, self.op.src)
    if op.op in LoadOps: return [(self.op, self, ())]
    if self.optype is MovementOps: return cast(LazyBuffer, self.op.src[0]).schedule(seen)

    if self.optype is BinaryOps: op = _ast_binaryops(op, self.shape)
    elif self.optype is ReduceOps:
      op = _ast_reduceops(op)
      if op.op in BinaryOps: op = _ast_binaryops(op, self.shape)

    # HACK: image shape can be wrong, hot cast it back to a normal float
    if isinstance(self.dtype, ImageDType) and (prod(self.shape) != prod(self.dtype.shape) or not any(self.shape[x]%4 == 0 for x in self.st.unit_stride_axes())):
      if op.op == MovementOps.RESHAPE: op = LazyOp(MovementOps.RESHAPE, (LazyOp(UnaryOps.CAST, op.src, (dtypes.float32, False)),), op.arg)
      else: op = LazyOp(UnaryOps.CAST, (op,), (dtypes.float32, False))
      self.dtype = dtypes.float32

    # realize the past and exec the AST
    ret = []
    for x in op.buffers: ret += x.schedule(seen)
    self.var_vals = dict(sorted(merge_dicts([buf.var_vals for buf in op.buffers]).items(), key=lambda kv:cast(Variable,kv[0]).key))

    # run the ast and log the op
    op, base_bufs = _replace_bufferops(op)
    return ret + [(op, self, base_bufs)]

  def realize(self:LazyBuffer) -> LazyBuffer:
    if not self.realized:
      for op,out,buffers in self.schedule():
        if DEBUG >= 3:
          from extra.utils import print_tree   # type: ignore
          print_tree(op)
        if op.op in LoadOps:
          LOAD_OPS_DISPATCHER[cast(LoadOps, op.op)](out)
        else:
          out.realized = Device[out.device].exec_ast(op, output=out, inputs=[x.realized for x in buffers], var_vals=out.var_vals, **self._device_extra_args())
          del out.op
        assert out.realized and isinstance(out.realized, (RawConst, Device[out.device].buffer)), f"device mismatch on realized got {type(out.realized)} expected {out.device}"
        assert out.realized.dtype == out.dtype, "realized dtype is incorrect"
    return self

  @staticmethod
  def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
    return create_lazybuffer(device, ShapeTracker.from_shape(tuple(shape)), LoadOps, LazyOp(op, tuple() if src is None else (src,), arg), dtype, {})

  # create a constant with the shape and dtype of self
  def const(self, val:Union[float, int]) -> LazyBuffer:
    # NOTE: dtypes.from_np(self.dtype.np) to deal with image types
    return self.loadop(LoadOps.CONST, tuple(), dtypes.from_np(self.dtype.np), self.device, arg=val).reshape((1,)*len(self.shape)).expand(self.shape)

  def contiguous(self:LazyBuffer) -> LazyBuffer:
    if not self.realized and self.op.op == LoadOps.CONTIGUOUS: return self  # two CONTIGUOUS in a row is one
    if self.st.contiguous and prod(self.shape) == prod(self.base.shape) and (hasattr(self.base, 'op') and self.base.op.op not in LoadOps) and (not self.realized or not isinstance(self.realized, RawConst)) and (not self.realized or prod(self.shape) == self.realized.size):
      # NOTE: the size can be wrong on LoadOps, so we excluded them all for now
      return self
    return create_lazybuffer(self.device, ShapeTracker.from_shape(self.shape), LoadOps, LazyOp(LoadOps.CONTIGUOUS, (self,), None), self.dtype, self.var_vals)

  @staticmethod
  def fromCPU(x: np.ndarray) -> LazyBuffer:
    return LazyBuffer("CPU", ShapeTracker.from_shape(x.shape), LoadOps, LazyOp(LoadOps.EMPTY, (), None), dtypes.from_np(x.dtype), {}, RawNumpyBuffer.fromCPU(x))

  def toCPU(self) -> np.ndarray:
    assert self.dtype.np, f"{self.dtype} is not supported in toCPU"
    self_casted = self.e(UnaryOps.CAST, arg=(dtypes.from_np(self.dtype.np), False)) if dtypes.from_np(self.dtype.np) != self.dtype else self
    realized = self_casted.contiguous().realize().realized
    assert all_int(self.shape), f"no toCPU if shape is symbolic, {self.shape=}"
    return cast(RawBuffer, realized).toCPU().reshape(self.shape)

  # *** elementwise ops ***

  def e(self:LazyBuffer, op:Union[UnaryOps, BinaryOps, TernaryOps], *srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
    # srcs includes self
    srcs = (self,)+srcs

    # if we are separated from other binary ops by movement ops, we push those movement ops above those binaryops
    if SHUFFLE_MOVEMENT_OPS: srcs = _push_movement_ops(srcs)

    # get outputs now
    out_device, out_shape, out_dtype = srcs[0].device, srcs[0].shape, max([x.dtype for x in srcs]) if op != UnaryOps.CAST else cast(Tuple[DType, bool], arg)[0]

    # push all contiguous to the end of BinaryOps. kernels 198 -> 196
    if PUSH_CONTIGUOUS and any(not x.realized and x.op.op == LoadOps.CONTIGUOUS and len(x.op.src[0].children) <= 1 for x in srcs):
      new_srcs: List[LazyBuffer] = []
      for x in srcs:
        if not x.realized and x.op.op == LoadOps.CONTIGUOUS and len(x.op.src[0].children) <= 1:
          x.op.src[0].children.discard(x)
          new_srcs.append(cast(LazyBuffer, x.op.src[0]))
        else:
          new_srcs.append(x)
      return new_srcs[0].e(op, *new_srcs[1:], arg=arg).contiguous()

    if MERGE_ELEMENTWISE_OPS:
      # remove the buffers from any (childless) BinaryOps that feed into this
      srcs = tuple([x.op if x.optype == BinaryOps and not x.children and not x.realized else x for x in srcs])  # type: ignore

    return create_lazybuffer(out_device, ShapeTracker.from_shape(out_shape), BinaryOps, LazyOp(op, srcs, arg), out_dtype, self.var_vals)

  # *** reduce ops ***

  def _reduce_op(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[sint, ...]) -> LazyBuffer:
    if self.shape == tuple(new_shape): return self
    srcs = _push_movement_ops((self,)) if SHUFFLE_MOVEMENT_OPS else (self,)
    return create_lazybuffer(self.device, ShapeTracker.from_shape(new_shape), ReduceOps, LazyOp(op, srcs, new_shape), self.dtype, self.var_vals)

  def r(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[sint, ...]) -> LazyBuffer:
    if any(not isinstance(s, int) for s in self.shape) or prod(self.shape) // prod(new_shape) < 32768: return self._reduce_op(op, new_shape) # The amount of work should be big enough to take the benefit of "2 kernels" approach.
    heuristic, divisor, dim_to_split = max(((divisor := math.gcd(256, old))/(stride or math.inf), divisor, i) for i, (old, new, stride) in enumerate(zip(self.shape, new_shape, self.st.real_strides())) if old != new) # type: ignore
    if divisor < 16 or heuristic < 0.1: return self._reduce_op(op, new_shape) # Choose largest divisor (>=16) to split on, penalize large strides.
    def splitted_shape(dim_aft_div): return self.shape[:dim_to_split] + (self.shape[dim_to_split]//divisor,) + dim_aft_div + self.shape[dim_to_split+1:]
    return self.reshape(splitted_shape((divisor,)))._reduce_op(op, splitted_shape((1,))).reshape(splitted_shape(()))._reduce_op(op, new_shape)

  # *** movement ops ***

  def shuffle_and_prune_movement_ops(self, st: ShapeTracker, op: MovementOps, arg: Union[Tuple[sint, ...], Tuple[Tuple[sint, sint], ...]]) -> LazyBuffer:
    if SHUFFLE_MOVEMENT_OPS and self.optype == BinaryOps and not self.realized and (op in {MovementOps.SHRINK, MovementOps.STRIDE, MovementOps.PERMUTE} or (op == MovementOps.RESHAPE and self.op.op in UnaryOps)) and not self.children:
      return self.op.replace_with_movement_ops([(op, arg)])
    if REMOVE_MOVEMENT_NOPS and not self.realized and st.contiguous:
      # MovementOps aren't stacked any more, they each have one parent, find the root
      root = get_movementroot(self)
      if root.st.contiguous and root != self and prod(st.shape) == prod(root.shape):
        return root.reshape(st.shape)
    return create_lazybuffer(self.device, st, MovementOps, LazyOp(op, (self,), arg), self.dtype, self.var_vals, base=self.base)

  def reshape(self:LazyBuffer, arg:Tuple[sint, ...]) -> LazyBuffer:
    if self.shape == arg: return self
    new_ints, new_nodes = partition(arg, lambda s: isinstance(s, int))
    if new_nodes and all(isinstance(s, int) for s in self.shape):
      # reshape from all int shape into shape with a variable, update the variable value
      assert len(new_nodes) == 1 and isinstance(new_nodes[0], Variable), "only support adding one Variable to the int shape"
      new_var, new_val = new_nodes[0], prod(self.shape) // prod(new_ints)
      # TODO: is it okay to set these var_vals on the base?
      if new_var not in self.var_vals:
        assert new_var.min <= new_val <= new_var.max, f"variable value {new_val} out of range [{new_var.min}, {new_var.max}]"
        self.var_vals[new_var] = new_val
      else: assert self.var_vals[new_var] == new_val, f"value conflicts, was {self.var_vals[new_var]}, set to {new_val}"
    if not self.realized and self.op.op == MovementOps.RESHAPE:
      assert isinstance(self.op.src[0], LazyBuffer)
      self.op.src[0].children.discard(self) # NOTE: this is only required in reshape and when pushing permutes, why??
      return self.op.src[0].reshape(arg)
    return self.shuffle_and_prune_movement_ops(self.st.reshape(arg), MovementOps.RESHAPE, arg)

  def pad(self:LazyBuffer, arg:Tuple[Tuple[int, int], ...]) -> LazyBuffer:
    if all(b == 0 and e == 0 for b,e in arg): return self
    if not self.realized and self.op.op == MovementOps.PAD: return self.op.src[0].pad(tuple([(b1+b2, e1+e2) for (b1,e1),(b2,e2) in zip(self.op.arg, arg)]))
    return self.shuffle_and_prune_movement_ops(self.st.pad(arg), MovementOps.PAD, arg)

  def expand(self: LazyBuffer, arg:Tuple[sint, ...]) -> LazyBuffer:
    if self.shape == arg: return self
    if not self.realized and self.op.op == MovementOps.EXPAND:
      return self.op.src[0].expand(arg)
    return self.shuffle_and_prune_movement_ops(self.st.expand(arg), MovementOps.EXPAND, arg)

  def permute(self: LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer:
    if arg == tuple(range(len(self.shape))): return self
    if not self.realized and self.op.op == MovementOps.PERMUTE: return self.op.src[0].permute(tuple([self.op.arg[i] for i in arg]))
    if not self.realized:
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
    return self.shuffle_and_prune_movement_ops(self.st.permute(arg), MovementOps.PERMUTE, arg)

  def shrink(self:LazyBuffer, arg:Tuple[Tuple[sint, sint], ...]) -> LazyBuffer:
    if all(b - a == s for s, (a, b) in zip(self.shape, arg)): return self
    if not self.realized and self.op.op == MovementOps.SHRINK: return self.op.src[0].shrink(tuple([(b1+b2, b1+e2) for (b1,_),(b2,e2) in zip(self.op.arg, arg)]))
    return self.shuffle_and_prune_movement_ops(self.st.shrink(arg), MovementOps.SHRINK, arg)

  def stride(self:LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer:
    if all(a == 1 for a in arg): return self
    if not self.realized and self.op.op == MovementOps.STRIDE: return self.op.src[0].stride(tuple(map(operator.mul, arg, self.op.arg)))
    return self.shuffle_and_prune_movement_ops(self.st.stride(arg), MovementOps.STRIDE, arg)

  def replace_with_movement_ops(self: LazyBuffer, ops:List[Tuple[MovementOps, Any]]) -> LazyBuffer:
    y = self
    for op, arg in ops: y = MOVEMENT_OPS_DISPATCHER[op](y, arg)
    return y

def _push_movement_ops(srcs:Tuple[LazyBuffer, ...]) -> Tuple[LazyBuffer, ...]:
  new_srcs = []
  for x in srcs:
    mops: List[Tuple[MovementOps, Any]] = []
    bx = x
    # backwalk all the movement ops. don't push PAD or EXPAND
    while not bx.realized and bx.optype is MovementOps and bx.op.op is not MovementOps.EXPAND and (SHUFFLE_PAD_OPS or bx.op.op is not MovementOps.PAD) and len(bx.children) <= 1:
      assert isinstance(bx.op.op, MovementOps)
      mops.append((bx.op.op, bx.op.arg))
      assert isinstance(bx.op.src[0], LazyBuffer)
      bx = bx.op.src[0]
    # NOTE: can't push pads past anything where f(0, 0) != 0 or f(0) != 0
    if mops and not bx.realized and bx.optype is BinaryOps and len(bx.children) <= 1 and (all(x[0] is not MovementOps.PAD for x in mops) or all(x.op not in UNSAFE_PAD_OPS for x in bx.op.get_lazyops())):
      new_srcs.append(bx.op.replace_with_movement_ops(mops[::-1]))
    else:
      new_srcs.append(x)
  return tuple(new_srcs)

MOVEMENT_OPS_DISPATCHER: Dict[MovementOps, Callable] = {
  MovementOps.RESHAPE: LazyBuffer.reshape,
  MovementOps.EXPAND: LazyBuffer.expand,
  MovementOps.SHRINK: LazyBuffer.shrink,
  MovementOps.PERMUTE: LazyBuffer.permute,
  MovementOps.PAD: LazyBuffer.pad,
  MovementOps.STRIDE: LazyBuffer.stride,
}

# *** loadop realization (unrelated to lazy) ***

def _realize_custom(buffer: LazyBuffer) -> None:
  # this needs to immediately realize
  buffer.realized = buffer.op.arg(buffer, *[x.realize() for x in buffer.op.src])

def _realize_from(buffer: LazyBuffer) -> None:
  rawbuf = buffer.op.src[0].realize()
  assert rawbuf.realized, "realize failed?"
  if DEBUG >= 3: print(f"*** copy {buffer.device} <- {rawbuf.device} size {rawbuf.realized.size} dtype {rawbuf.realized.dtype}")
  # TODO: make this generic
  if isinstance(rawbuf.realized, RawDiskBuffer) and issubclass(Device[buffer.device].buffer, RawBufferMapped):
    assert all_int(buffer.shape), "does not support symbolic shape"
    buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())
    rawbuf.realized.readinto(cast(RawBufferMapped, buffer.realized)._buffer())
  elif isinstance(rawbuf.realized, RawBufferTransfer) and issubclass(Device[buffer.device].buffer, RawBufferTransfer) and P2P >= 1:
    buffer.realized = cast(RawBufferTransfer, Device[buffer.device].buffer).transfer(rawbuf.realized, buffer.shape, buffer.dtype, **buffer._device_extra_args())
  else:
    buffer.realized = Device[buffer.device].buffer.fromCPU(rawbuf.toCPU(), **buffer._device_extra_args())

def _realize_empty(buffer: LazyBuffer) -> None:
  assert all_int(buffer.shape), "does not support symbolic shape"
  buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())

def _realize_rand(buffer: LazyBuffer) -> None:
  rng = np.random.default_rng(buffer.op.arg)
  buffer.realized = Device[buffer.device].buffer.fromCPU(rng.random(size=buffer.shape, dtype=np.float32).astype(dtype=buffer.dtype.np, copy=False), **buffer._device_extra_args()) # type: ignore

def _realize_const(buffer: LazyBuffer) -> None:
  if isinstance(Device[buffer.device], Compiled) and buffer.device not in ["LLVM"]:  # consts are broken in LLVM in NaN/inf
    buffer.realized = RawConst(1, buffer.dtype, float(buffer.op.arg))
  else:
    buffer.realized = Device[buffer.device].buffer.fromCPU(np.array(buffer.op.arg, dtype=buffer.dtype.np), **buffer._device_extra_args())

LOAD_OPS_DISPATCHER: Dict[LoadOps, Callable] = {
  LoadOps.CUSTOM: _realize_custom,
  LoadOps.FROM: _realize_from,
  LoadOps.EMPTY: _realize_empty,
  LoadOps.RAND: _realize_rand,
  LoadOps.CONST: _realize_const,
}
