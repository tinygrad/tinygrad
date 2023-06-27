from __future__ import annotations
import operator
from typing import Callable, Optional, Tuple, Union, List, Dict, Any, cast
import sys, importlib, inspect, functools, pathlib
from weakref import ref

import numpy as np
from tinygrad.helpers import GRAPH, DEBUG, prod, getenv, DType, dtypes, flatten, ImageDType, LightWeakSet, LightWeakValueDictionary
from tinygrad.runtime.ops_cpu import RawNumpyBuffer
from tinygrad.runtime.ops_disk import RawDiskBuffer
from tinygrad.shape.shapetracker import MovementOps, ShapeTracker, get_contraction
from tinygrad.ops import Compiled, Interpreted, UnaryOps, BinaryOps, ReduceOps, LoadOps, OpType, LazyOp
from tinygrad.runtime.lib import RawBufferMapped, RawConst, RawBuffer

# lazy can recurse a lot
sys.setrecursionlimit(10000)

OPT = getenv("OPT", 2)
LAZY = getenv("LAZY", 1)
LAZYCACHE = getenv("LAZYCACHE", 1)

# TODO: movement ops that only change shape are really nops. treat them as such
REMOVE_MOVEMENT_NOPS, MERGE_ELEMENTWISE_INTO_REDUCE, SHUFFLE_MOVEMENT_OPS, MERGE_ELEMENTWISE_OPS = OPT>=1, OPT>=1, OPT>=1, OPT>=1
MERGE_ONE_REDUCE_INTO_ELEMENTWISE, SHUFFLE_PAD_OPS = OPT>=2, OPT>=2   # shuffle pad ops is fine now since we only push to merge binops
PUSH_PERMUTES, PUSH_CONTIGUOUS = OPT>=3, OPT>=3

# **** realize functions ****
def _ast_reduceops(self:LazyBuffer) -> LazyOp:
  # TODO: this can also corealize a binary op after the reduce, not just before
  src = self.op.src[0]
  if MERGE_ELEMENTWISE_INTO_REDUCE and not src.realized and src.optype is BinaryOps and len(src.children) <= 1:
    src = src.op # type: ignore
  return LazyOp(self.op.op, (src,), self.op.arg)

# this supports late merging an upstream Reduce op and even an Elementwise op above that
def _ast_binaryops(self:LazyBuffer) -> LazyOp:
  real_srcs: Dict[LazyBuffer, Union[None, LazyOp, LazyBuffer]] = {x:None for x in self.op.buffers}
  # NOTE: contiguous does not always mean the same size with SHRINK. this is still mergeable but requires more thought how
  # TODO: this can also support late fusion of BinaryOps, required for test_fold_conv_sgd
  psrcs: List[Tuple[LazyBuffer, LazyBuffer]] = [(k,x) for k,x in zip(real_srcs.keys(), map(get_movementroot_contiguous, real_srcs.keys())) if x.optype == ReduceOps and not x.realized and prod(k.shape) == prod(x.shape) and len(x.children) <= 1 and len(k.children) <= 1]
  intermediate_shape: Tuple[int, ...] = self.shape
  if MERGE_ONE_REDUCE_INTO_ELEMENTWISE and len(psrcs) >= 1:
    psrc = psrcs[0] # NOTE: right now we can't handle multiple, as we'd have to check for loop
    if psrc[1].optype == ReduceOps:
      top = _ast_reduceops(psrc[1])
    real_srcs[psrc[0]] = top
    real_srcs.update({x:x for x in top.buffers})  # the reduce op buffers are not modified

    # if the ReduceOp is followed by a reshape, we push this reshape before all the ElementwiseOp inputs
    if psrc[0].shape != psrc[1].shape:
      intermediate_shape = psrc[1].shape
      assert psrc[0].shape == self.shape, f"shape mismatch {psrc[0].shape} != {self.shape}"

  # reshape all the late ops into the output shape
  # NOTE: these RESHAPEs will return self if they don't change the shape
  for x in real_srcs.keys():
    if not real_srcs[x]: real_srcs[x] = x.reshape(intermediate_shape)
  ast = self.op.map_buffers(real_srcs)
  return LazyOp(MovementOps.RESHAPE, (ast, ), self.shape) if intermediate_shape != self.shape else ast

# **** lazy operations ****

def get_single_root(root:LazyBuffer) -> LazyBuffer: return get_single_root(cast(LazyBuffer, root.op.src[0])) if getattr(root, 'op', None) and len(root.op.src) == 1 else root
def get_movementroot(root:LazyBuffer, allow_contiguous=False) -> LazyBuffer: return get_movementroot(cast(LazyBuffer, root.op.src[0]), allow_contiguous) if not root.realized and (root.optype == MovementOps or (root.op.op == LoadOps.CONTIGUOUS and allow_contiguous and root.op.src[0].st.contiguous)) else root
def get_movementroot_contiguous(x:LazyBuffer) -> LazyBuffer: return get_movementroot_contiguous(cast(LazyBuffer, x.op.src[0])) if not x.realized and x.op.op == LoadOps.CONTIGUOUS else (get_movementroot(x, True) if x.optype == MovementOps and x.st.contiguous else x)

lazycache: LightWeakValueDictionary = LightWeakValueDictionary()
def create_lazybuffer(device:str, st:ShapeTracker, optype:OpType, op:LazyOp, dtype:DType):
  #print("create_lazybuffer", device, shape, optype, op, dtype)

  # fromcpu aren't cached
  if not LAZYCACHE or (optype is LoadOps and op.op in {LoadOps.EMPTY, LoadOps.RAND, LoadOps.CONST}): return LazyBuffer(device, st, optype, op, dtype)

  # wop is the deduping key. i feel this used to compare more deeply
  wop = (device, dtype, optype, ref(op))
  if wop in lazycache: return lazycache[wop]

  lazycache[wop] = ret = LazyBuffer(device, st, optype, op, dtype)
  return ret

class LazyBuffer:
  __slots__ = 'st', 'device', 'shape', 'optype', 'dtype', 'op', 'realized', 'output_buffer', 'children', 'node_id', '__weakref__'
  __deletable__ = ('op',)
  def __init__(self, device:str, st:ShapeTracker, optype:OpType, op:LazyOp, dtype:DType, src:Optional[RawBuffer]=None):
    self.st: ShapeTracker = st  # NOTE: this is not a copy! this should be a "read-only" ShapeTracker
    self.device, self.shape, self.optype, self.dtype = device, self.st.shape, optype, dtype
    self.realized: Optional[RawBuffer] = src
    self.output_buffer: Optional[RawBuffer] = None   # TODO: do we really need this? or can we just use realized
    # TODO: does children have to be a ref count instead of a set? can a Buffer be a double child?
    self.children: LightWeakSet = LightWeakSet()
    # NOTE: op should be read only after construction of LazyBuffer
    self.op: LazyOp = op
    for x in op.buffers: x.children.add(self)
    if not LAZY: self.realize()

    # log phantom ops to the graph
    if GRAPH >= 3:
      from tinygrad.graph import log_op
      log_op(self, self.op, phantom=True)

  def __repr__(self): return f"<LB {self.shape} {self.dtype} op={self.op.op if not self.realized else self.realized} st={self.st}>"
  @property
  def key(self):
    if self.realized: return (self.dtype.key, self.realized.key, self.st.key)
    return (self.dtype.key, self.op.op, self.st.key)

  def _device_extra_args(self) -> Dict[str, str]: return {"device": self.device.split(":", 1)[1]} if ":" in self.device else {}

  def realize(self:LazyBuffer) -> LazyBuffer:
    if not self.realized:
      # get real ops first
      if self.optype is BinaryOps: self.op = _ast_binaryops(self)
      elif self.optype is ReduceOps: self.op = _ast_reduceops(self)
      elif self.optype is LoadOps: LOAD_OPS_DISPATCHER[cast(LoadOps, self.op.op)](self)
      # run the ast if we still have to, and log the op
      if not self.realized:
        for x in self.op.buffers: x.realize()

        # HACK: image shape can be wrong, hot cast it back to a normal float
        if self.dtype.__class__ is ImageDType and self.optype != MovementOps and (prod(self.shape) != prod(cast(ImageDType, self.dtype).shape) or not any([self.shape[x]%4 == 0 for x in self.st.unit_stride_axes()])):
          if self.op.op == MovementOps.RESHAPE:
            # put CAST before the final RESHAPE
            self.op = LazyOp(MovementOps.RESHAPE, (LazyOp(UnaryOps.CAST, self.op.src, dtypes.float32),), self.op.arg)
          else:
            self.op = LazyOp(UnaryOps.CAST, (self.op,), dtypes.float32)
          self.dtype = dtypes.float32
        self.realized = Device[self.device].exec_ast(self.op, output=self, **self._device_extra_args())

      assert self.realized and isinstance(self.realized, (RawConst, Device[self.device].buffer)), f"device mismatch on realized got {type(self.realized)} expected {self.device}"
      # HACK: allow hot casting of images
      assert self.realized.dtype == self.dtype or self.dtype.__class__ is ImageDType, f"dtype mismatch on realize got {self.realized.dtype} expected {self.dtype}"
      self.dtype = self.realized.dtype

      # log to the graph
      if (DEBUG or GRAPH) and (self.realized.__class__ is not RawConst or GRAPH >= 2):
        from tinygrad.graph import log_op
        log_op(self, self.op)

      # no need to keep the op after realization
      del self.op
    return self

  @staticmethod
  def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
    return create_lazybuffer(device, ShapeTracker(tuple(shape)), LoadOps, LazyOp(op, tuple() if src is None else (src,), arg), dtype)

  @staticmethod
  def fromCPU(x: np.ndarray) -> LazyBuffer:
    return LazyBuffer("CPU", ShapeTracker(x.shape), LoadOps, LazyOp(LoadOps.EMPTY, (), None), dtypes.from_np(x.dtype), RawNumpyBuffer.fromCPU(x))

  # create a constant with the shape and dtype of self
  def const_like(self, val) -> LazyBuffer:
    # NOTE: dtypes.from_np(self.dtype.np) to deal with image types
    return self.loadop(LoadOps.CONST, tuple(), dtypes.from_np(self.dtype.np), self.device, arg=val).reshape((1,)*len(self.shape)).expand(self.shape)

  # NOTE: we also have to copy the numpy array on the way out...otherwise the underlying Tensor could be freed and use after free. improve this?
  def toCPU(self):
    realized = self.cast(dtypes.from_np(self.dtype.np)).contiguous().realize().realized
    ret = cast(RawBuffer, realized).toCPU().reshape(self.shape)
    return ret

  def cast(self:LazyBuffer, arg:DType) -> LazyBuffer: return elementwise_op(UnaryOps.CAST, self, arg=arg) if self.dtype != arg else self
  def unary_op(self:LazyBuffer, op:UnaryOps) -> LazyBuffer: return elementwise_op(op, self)
  def binary_op(self:LazyBuffer, op:BinaryOps, y:LazyBuffer) -> LazyBuffer: return elementwise_op(op, self, y)
  def contiguous(self:LazyBuffer) -> LazyBuffer:
    if not self.realized and self.op.op == LoadOps.CONTIGUOUS: return self  # two CONTIGUOUS in a row is one
    return create_lazybuffer(self.device, ShapeTracker(self.shape), LoadOps, LazyOp(LoadOps.CONTIGUOUS, (self,), None), self.dtype)

  def shuffle_and_prune_movement_ops(self, st: ShapeTracker, op: MovementOps, arg: Union[Tuple[int, ...], Tuple[Tuple[int, int], ...]]) -> LazyBuffer:
    if SHUFFLE_MOVEMENT_OPS and self.optype == BinaryOps and not self.realized and (op in {MovementOps.SHRINK, MovementOps.STRIDE, MovementOps.PERMUTE} or (op == MovementOps.RESHAPE and self.op.op in UnaryOps)) and len(self.children) == 0:
      return self.op.replace_with_movement_ops([(op, arg)])
    ret = create_lazybuffer(self.device, st, MovementOps, LazyOp(op, (self,), arg), self.dtype)
    if REMOVE_MOVEMENT_NOPS and not self.realized and not ret.realized and ret.st.contiguous:
      # MovementOps aren't stacked any more, they each have one parent, find the root
      root = get_movementroot(self)
      if root.st.contiguous and root != self and prod(ret.st.shape) == prod(root.shape):
        return root.reshape(ret.st.shape)
    return ret

  def reduce_op(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[int, ...]) -> LazyBuffer:
    if self.shape == tuple(new_shape): return self
    srcs = _push_movement_ops((self,)) if SHUFFLE_MOVEMENT_OPS else (self,)
    return create_lazybuffer(self.device, ShapeTracker(new_shape), ReduceOps, LazyOp(op, srcs, new_shape), self.dtype)

  def reshape(self:LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer:
    if self.shape == arg: return self
    if not self.realized and self.op.op == MovementOps.RESHAPE:
      self.op.src[0].children.discard(self) # NOTE: this is only required in reshape and when pushing permutes, why??
      return self.op.src[0].reshape(arg)
    return self.shuffle_and_prune_movement_ops(ShapeTracker(self.st).reshape(arg), MovementOps.RESHAPE, arg)

  def pad(self:LazyBuffer, arg:Tuple[Tuple[int, int], ...]) -> LazyBuffer:
    if all([b == 0 and e == 0 for b,e in arg]): return self
    if not self.realized and self.op.op == MovementOps.PAD: return self.op.src[0].pad(tuple([(b1+b2, e1+e2) for (b1,e1),(b2,e2) in zip(self.op.arg, arg)]))
    return self.shuffle_and_prune_movement_ops(ShapeTracker(self.st).pad(arg), MovementOps.PAD, arg)

  def expand(self: LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer:
    if self.shape == arg: return self
    if not self.realized and self.op.op == MovementOps.EXPAND:
      return self.op.src[0].expand(arg)
    return self.shuffle_and_prune_movement_ops(ShapeTracker(self.st).expand(arg), MovementOps.EXPAND, arg)

  def permute(self: LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer:
    if arg == tuple(range(len(self.shape))): return self
    if not self.realized and self.op.op == MovementOps.PERMUTE: return self.op.src[0].permute(tuple([self.op.arg[i] for i in arg]))
    if not self.realized:
      if PUSH_PERMUTES and self.optype == ReduceOps:
        # reduceops have one buffer input, permute it
        narg = tuple([self.op.arg[arg[i]] for i in range(len(arg))])
        src, rop = self.op.src[0], self.op.op
        src.children.discard(self)
        del self  # TODO: why doesn't this delete remove it from the children
        return src.permute(arg).reduce_op(cast(ReduceOps, rop), narg)

      # move permutes before expands (always, this is safe)
      if self.op.op == MovementOps.EXPAND:
        return self.op.src[0].permute(arg).expand(tuple([self.op.arg[a] for a in arg]))

      # move permutes before reshapes if we can
      if PUSH_PERMUTES and self.op.op == MovementOps.RESHAPE and self.op.src[0].__class__ is LazyBuffer:
        if shape_idx_groups := get_contraction(self.op.src[0].shape, self.shape):
          self.op.src[0].children.discard(self) # NOTE: this is only required in reshape and when pushing permutes, why??
          return self.op.src[0].permute(tuple(flatten(shape_idx_groups[i] for i in arg))).reshape(ShapeTracker(self.st).permute(arg).shape)
    return self.shuffle_and_prune_movement_ops(ShapeTracker(self.st).permute(arg), MovementOps.PERMUTE, arg)

  def shrink(self:LazyBuffer, arg:Tuple[Tuple[int, int], ...]) -> LazyBuffer:
    if all([b - a == s for s, (a, b) in zip(self.shape, arg)]): return self
    if not self.realized and self.op.op == MovementOps.SHRINK: return self.op.src[0].shrink(tuple([(b1+b2, b1+e2) for (b1,_),(b2,e2) in zip(self.op.arg, arg)]))
    return self.shuffle_and_prune_movement_ops(ShapeTracker(self.st).shrink(arg), MovementOps.SHRINK, arg)

  def stride(self:LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer:
    local_st = ShapeTracker(self.shape).stride(arg)
    if self.shape == local_st.shape and local_st.contiguous: return self
    if not self.realized and self.op.op == MovementOps.STRIDE: return self.op.src[0].stride(tuple(map(operator.mul, arg, self.op.arg)))
    return self.shuffle_and_prune_movement_ops(ShapeTracker(self.st).stride(arg), MovementOps.STRIDE, arg)

  @property
  def buffers(self) -> Tuple[LazyBuffer, ...]: return (self,)
  def map_buffers(self, real_srcs: Dict[Any, Any]): return real_srcs.get(self, self)
  def get_lazyops(self) -> List[Any]: return []
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
    while not bx.realized and bx.optype == MovementOps and bx.op.op != MovementOps.EXPAND and (SHUFFLE_PAD_OPS or bx.op.op != MovementOps.PAD) and len(bx.children) <= 1:
      assert isinstance(bx.op.op, MovementOps)
      mops.append((bx.op.op, bx.op.arg))
      bx = cast(LazyBuffer, bx.op.src[0])
    # NOTE: can't push pads with a div
    if not bx.realized and bx.optype == BinaryOps and len(bx.children) <= 1 and len(mops) and (all([x[0] != MovementOps.PAD for x in mops]) or all([x.op != BinaryOps.DIV for x in bx.op.get_lazyops()])):
      new_srcs.append(bx.op.replace_with_movement_ops(mops[::-1]))
    else:
      new_srcs.append(x)
  return tuple(new_srcs)

def elementwise_op(op:Union[UnaryOps, BinaryOps], *srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
  # if we are separated from other binary ops by movement ops, we push those movement ops above those binaryops
  if SHUFFLE_MOVEMENT_OPS: srcs = _push_movement_ops(srcs)

  # get outputs now
  out_device, out_shape, out_dtype = srcs[0].device, srcs[0].shape, max([x.dtype for x in srcs]) if op != UnaryOps.CAST else cast(DType, arg)

  # push all contiguous to the end of BinaryOps. kernels 198 -> 196
  if PUSH_CONTIGUOUS and any([not x.realized and x.op.op == LoadOps.CONTIGUOUS and len(x.op.src[0].children) <= 1 for x in srcs]):
    new_srcs: List[LazyBuffer] = []
    for x in srcs:
      if not x.realized and x.op.op == LoadOps.CONTIGUOUS and len(x.op.src[0].children) <= 1:
        x.op.src[0].children.discard(x)
        new_srcs.append(cast(LazyBuffer, x.op.src[0]))
      else:
        new_srcs.append(x)
    return elementwise_op(op, *new_srcs, arg=arg).contiguous()

  if MERGE_ELEMENTWISE_OPS:
    # remove the buffers from any (childless) BinaryOps that feed into this
    srcs = tuple([x.op if x.optype == BinaryOps and len(x.children) == 0 and not x.realized else x for x in srcs])  # type: ignore

  return create_lazybuffer(out_device, ShapeTracker(out_shape), BinaryOps, LazyOp(op, srcs, arg), out_dtype)

class _Device:
  def __init__(self) -> None:
    self._buffers: List[str] = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
    self.DEFAULT: str = functools.reduce(lambda val, ele: ele if getenv(ele) == 1 else val, self._buffers, None) or self._default_device()
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def canonicalize(self, device:Optional[str]) -> str: return (device.split(":", 1)[0].upper() + ((":"+device.split(":", 1)[1]) if ':' in device else '')).replace(":0", "") if device is not None else self.DEFAULT
  def __getitem__(self, x:str) -> Union[Interpreted, Compiled]: return self._get_device(x.split(":")[0].upper())
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def _get_device(self, x:str) -> Union[Interpreted, Compiled]: return [cls for cname, cls in inspect.getmembers(importlib.import_module(f'tinygrad.runtime.ops_{x.lower()}')) if (cname.lower() == x.lower() + "buffer") and x in self._buffers][0]
  def _default_device(self) -> str:
    for device in ["METAL", "CUDA", "GPU"]:
      try:
        if self[device]: return device
      except Exception: pass
    return "CPU"
Device = _Device()

def _realize_contiguous(buffer: LazyBuffer) -> None:
  realized = buffer.op.src[0].realize().realized
  if buffer.op.src[0].st.contiguous and realized.__class__ is not RawConst and cast(RawBuffer, realized).size == prod(buffer.shape):
    # no need to run an AST, this is already contiguous
    buffer.realized = realized
  else:
    # TODO: remove UnaryOps.NOOP, replace with LoadOps.CONTIGUOUS. confusing with Compiled though
    buffer.op = LazyOp(UnaryOps.NOOP, buffer.op.src)

def _realize_custom(buffer: LazyBuffer) -> None:
  # this needs to immediately realize
  buffer.realized = buffer.op.arg(buffer, *[x.realize() for x in buffer.op.src])

def _realize_from(buffer: LazyBuffer) -> None:
  rawbuf = buffer.op.src[0].realize()
  # TODO: make this generic
  if isinstance(rawbuf.realized, RawDiskBuffer) and issubclass(Device[buffer.device].buffer, RawBufferMapped):
    buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())
    rawbuf.realized.readinto(cast(RawBufferMapped, buffer.realized)._buffer())
  else:
    buffer.realized = Device[buffer.device].buffer.fromCPU(rawbuf.toCPU(), **buffer._device_extra_args())

def _realize_empty(buffer: LazyBuffer) -> None:
  buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())

def _realize_rand(buffer: LazyBuffer) -> None:
  rng = np.random.default_rng(buffer.op.arg)
  buffer.realized = Device[buffer.device].buffer.fromCPU(rng.random(size=buffer.shape, dtype=buffer.dtype.np), **buffer._device_extra_args()) # type: ignore

def _realize_const(buffer: LazyBuffer) -> None:
  if hasattr(Device[buffer.device].codegen, 'supports_constant_folding'):
    buffer.realized = RawConst(1, buffer.dtype, float(buffer.op.arg))
  else:
    buffer.realized = Device[buffer.device].buffer.fromCPU(np.array(buffer.op.arg, dtype=buffer.dtype.np), **buffer._device_extra_args())

LOAD_OPS_DISPATCHER: Dict[LoadOps, Callable] = {
  LoadOps.CONTIGUOUS: _realize_contiguous,
  LoadOps.CUSTOM: _realize_custom,
  LoadOps.FROM: _realize_from,
  LoadOps.EMPTY: _realize_empty,
  LoadOps.RAND: _realize_rand,
  LoadOps.CONST: _realize_const,
}

MOVEMENT_OPS_DISPATCHER: Dict[MovementOps, Callable] = {
  MovementOps.RESHAPE: LazyBuffer.reshape,
  MovementOps.EXPAND: LazyBuffer.expand,
  MovementOps.SHRINK: LazyBuffer.shrink,
  MovementOps.PERMUTE: LazyBuffer.permute,
  MovementOps.PAD: LazyBuffer.pad,
  MovementOps.STRIDE: LazyBuffer.stride,
}