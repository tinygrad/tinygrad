from __future__ import annotations
from typing import Optional, Tuple, Union, List, Dict, Any, cast
import sys, importlib, inspect, functools, pathlib
from weakref import WeakSet, WeakValueDictionary, ref
from tinygrad.helpers import getenv, DType, dtypes, LazyNumpyArray, flatten, ImageDType, DEBUG
from math import prod
from tinygrad.shape.shapetracker import MOVEMENT_OPS, ShapeTracker, get_contraction
from tinygrad.ops import Compiled, Interpreted, UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, OpType, LazyOp
from tinygrad.runtime.lib import RawConst, RawBuffer
from tinygrad.graph import log_op, GRAPH

# lazy can recurse a lot
sys.setrecursionlimit(10000)

OPT = getenv("OPT", 2)
LAZY = getenv("LAZY", 1)

# TODO: movement ops that only change shape are really nops. treat them as such
REMOVE_MOVEMENT_NOPS, MERGE_ELEMENTWISE_INTO_REDUCE, SHUFFLE_MOVEMENT_OPS, MERGE_ELEMENTWISE_OPS = OPT>=1, OPT>=1, OPT>=1, OPT>=1
MERGE_ONE_REDUCE_INTO_ELEMENTWISE, SHUFFLE_PAD_OPS = OPT>=2, OPT>=2   # shuffle pad ops is fine now since we only push to merge binops
PUSH_PERMUTES, PUSH_CONTIGUOUS = OPT>=3, OPT>=3

# **** realize functions ****
def _ast_reduceops(self:LazyBuffer) -> LazyOp:
  # TODO: this can also corealize a binary op after the reduce, not just before
  src = self.op.src[0]
  if MERGE_ELEMENTWISE_INTO_REDUCE and not src.realized and src.optype == BinaryOps and len(src.children) <= 1:
    src = src.op
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
    if not real_srcs[x]: real_srcs[x] = x.reshape_op(intermediate_shape)
  ast = self.op.map_buffers(real_srcs)
  return LazyOp(MovementOps.RESHAPE, (ast, ), self.shape) if intermediate_shape != self.shape else ast

# **** lazy operations ****

def get_single_root(root:LazyBuffer) -> LazyBuffer: return get_single_root(root.op.src[0]) if getattr(root, 'op', None) and len(root.op.src) == 1 else root
def get_movementroot(root:LazyBuffer, allow_contiguous=False) -> LazyBuffer: return get_movementroot(root.op.src[0], allow_contiguous) if not root.realized and (root.optype == MovementOps or (root.op.op == LoadOps.CONTIGUOUS and allow_contiguous and root.op.src[0].st.contiguous)) else root
def get_movementroot_contiguous(x:LazyBuffer) -> LazyBuffer: return get_movementroot_contiguous(x.op.src[0]) if not x.realized and x.op.op == LoadOps.CONTIGUOUS else (get_movementroot(x, True) if x.optype == MovementOps and x.st.contiguous else x)

lazycache: WeakValueDictionary[Tuple[str, DType, OpType, LazyOp], LazyBuffer] = WeakValueDictionary()
def create_lazybuffer(device:str, shape:Union[ShapeTracker, Tuple[int, ...]], optype:OpType, op:LazyOp, dtype:DType):
  st = shape if shape.__class__ == ShapeTracker else ShapeTracker(tuple(shape))

  # fromcpu aren't cached
  if optype == LoadOps and op.op in {LoadOps.FROMCPU, LoadOps.EMPTY}: return LazyBuffer(device, st, optype, op, dtype)

  #print("create_lazybuffer", device, shape, optype, op, dtype)

  # NOTE: shape should be deterministic. annoying to cache with the ShapeTracker
  # get_weakop makes all the LazyBuffers in the op have a weakref
  wop = (device, dtype, optype, ref(op))

  if wop not in lazycache: lazycache[wop] = ret = LazyBuffer(device, st, optype, op, dtype)
  else: ret = lazycache[wop]
  return ret

class LazyBuffer:
  __slots__ = 'st', 'device', 'shape', 'optype', 'dtype', 'op', 'realized', 'output_buffer', 'children', 'node_id', '__weakref__'
  __deletable__ = ('op',)
  def __init__(self, device:str, st:ShapeTracker, optype:OpType, op:LazyOp, dtype:DType):
    self.st = st  # NOTE: this is not a copy! this should be a "read-only" ShapeTracker
    self.device, self.shape, self.optype, self.dtype = device, self.st.shape, optype, dtype
    self.op: LazyOp = op
    self.realized: Optional[RawBuffer] = None
    self.output_buffer: Optional[RawBuffer] = None   # TODO: do we really need this? or can we just use realized
    # TODO: does children have to be a ref count instead of a set? can a Buffer be a double child?
    self.children: WeakSet[LazyBuffer] = WeakSet()
    # NOTE: op should be read only after construction of LazyBuffer
    for x in op.buffers: x.children.add(self)
    if not LAZY: self.realize()

    # log phantom ops to the graph
    if GRAPH >= 3: log_op(self, self.op, phantom=True)

  def __repr__(self): return f"<LB {self.shape} {self.dtype} op:{self.op.op if not self.realized else self.realized} st:{self.st}>"
  def _device_extra_args(self) -> Dict[str, str]: return {"device": self.device.split(":")[1]} if ":" in self.device else {}

  def realize(self:LazyBuffer) -> LazyBuffer:
    if not self.realized:
      # get real ops first
      if self.optype == ReduceOps: self.op = _ast_reduceops(self)
      elif self.optype == BinaryOps: self.op = _ast_binaryops(self)  # ISSUE: this can include a reshape
      else:
        try: REALIZE_DISPATCHER[self.op.op](self)
        except KeyError: pass
      # run the ast if we still have to, and log the op
      if not self.realized:
        for x in self.op.buffers: x.realize()

        # HACK: image shape can be wrong, hot cast it back to a normal float
        if self.optype != MovementOps and self.dtype.__class__ == ImageDType and (prod(self.shape) != prod(self.dtype.shape) or not any([self.shape[x]%4 == 0 for x in self.st.unit_stride_axes()])):
          if self.op.op == MovementOps.RESHAPE:
            # put CAST before the final RESHAPE
            self.op = LazyOp(MovementOps.RESHAPE, (LazyOp(UnaryOps.CAST, self.op.src, dtypes.float32),), self.op.arg)
          else:
            self.op = LazyOp(UnaryOps.CAST, (self.op,), dtypes.float32)
          self.dtype = dtypes.float32

        self.realized = Device[self.device].exec_ast(self.op, output=self, **self._device_extra_args())

      assert self.realized.__class__ in {RawConst, Device[self.device].buffer}, f"device mismatch on realized got {type(self.realized)} expected {self.device}"
      # HACK: allow hot casting of images
      assert self.realized.dtype == self.dtype or self.dtype.name.startswith("image"), f"dtype mismatch on realize got {self.realized.dtype} expected {self.dtype}"
      self.dtype = self.realized.dtype

      # log to the graph
      if not self.realized.__class__ == RawConst or GRAPH >= 2: log_op(self, self.op)

      # no need to keep the op after realization
      del self.op
    return self

  # NOTE: we have to make a copy of the numpy array here in case the user changes it. expose this? LazyNumpyArray doesn't have this problem
  @staticmethod
  def fromCPU(x:LazyNumpyArray, device) -> LazyBuffer:
    return create_lazybuffer(device, x.shape, LoadOps, LazyOp(LoadOps.FROMCPU, tuple(), x), dtypes.from_np(x.dtype))

  @staticmethod
  def empty(shape, dtype, device) -> LazyBuffer:
    return create_lazybuffer(device, shape, LoadOps, LazyOp(LoadOps.EMPTY, tuple()), dtype)

  # create a constant with the shape and dtype of self
  def const_like(self, val) -> LazyBuffer:
    return create_lazybuffer(self.device, (1,), LoadOps, LazyOp(LoadOps.FROMCPU, tuple(), LazyNumpyArray([val], (1,), self.dtype.np)), self.dtype) \
      .reshape_op((1,)*len(self.shape)).expand_op(self.shape)

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
    return create_lazybuffer(self.device, self.shape, LoadOps, LazyOp(LoadOps.CONTIGUOUS, (self,)), self.dtype)

  def reduce_op(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[int, ...]) -> LazyBuffer:
    if self.shape == tuple(new_shape): return self
    srcs = _push_movement_ops((self,)) if SHUFFLE_MOVEMENT_OPS else (self,)
    return create_lazybuffer(self.device, new_shape, ReduceOps, LazyOp(op, tuple(srcs), new_shape), self.dtype)

  def reshape_op(self:LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer:
    shape, realized = self.shape, self.realized
    if shape == arg: return self
    if not realized and self.op.op == MovementOps.RESHAPE:
      self.op.src[0].children.discard(self)
      return self.op.src[0].reshape_op(arg)
    if SHUFFLE_MOVEMENT_OPS and self.optype == BinaryOps and not realized and self.op.op in UnaryOps and len(self.children) == 0:
      return self.op.replace_with_movement_ops([(MovementOps.RESHAPE, arg)])
    ret = create_lazybuffer(self.device, ShapeTracker(self.st).reshape(arg), MovementOps, LazyOp(MovementOps.RESHAPE, (self,), arg), self.dtype)
    if REMOVE_MOVEMENT_NOPS and not ret.realized and not realized and ret.st.contiguous:
      # MovementOps aren't stacked any more, they each have one parent, find the root
      root = get_movementroot(self)
      if root.st.contiguous and root != self and prod(ret.st.shape) == prod(root.shape):
        return root.reshape_op(ret.st.shape)

    return ret

  def pad_op(self:LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer:
    realized = self.realized
    if all([b == 0 and e == 0 for b,e in arg]): return self
    if not realized and self.op.op == MovementOps.PAD:
      self.op.src[0].children.discard(self)
      return self.op.src[0].pad_op(tuple([(b1+b2, e1+e2) for (b1,e1),(b2,e2) in zip(self.op.arg, arg)]))
    ret = create_lazybuffer(self.device, ShapeTracker(self.st).pad(arg), MovementOps, LazyOp(MovementOps.PAD, (self,), arg), self.dtype)
    if REMOVE_MOVEMENT_NOPS and not ret.realized and not realized and ret.st.contiguous:
      # MovementOps aren't stacked any more, they each have one parent, find the root
      root = get_movementroot(self)
      if root.st.contiguous and root != self and prod(ret.st.shape) == prod(root.shape):
        return root.reshape_op(ret.st.shape)
    return ret

  def expand_op(self: LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer:
    shape, realized = self.shape, self.realized
    if shape == arg: return self
    if not realized and self.op.op == MovementOps.EXPAND:
      self.op.src[0].children.discard(self)
      return self.op.src[0].expand_op(arg)
    ret = create_lazybuffer(self.device, ShapeTracker(self.st).expand(arg), MovementOps, LazyOp(MovementOps.EXPAND, (self,), arg), self.dtype)
    if REMOVE_MOVEMENT_NOPS and not ret.realized and not realized and ret.st.contiguous:
      # MovementOps aren't stacked any more, they each have one parent, find the root
      root = get_movementroot(self)
      if root.st.contiguous and root != self and prod(ret.st.shape) == prod(root.shape):
        return root.reshape_op(ret.st.shape)
    return ret

  def permute_op(self: LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer:
    shape, realized = self.shape, self.realized
    if arg == tuple(range(len(shape))): return self
    if not realized and self.op.op == MovementOps.PERMUTE:
      self.op.src[0].children.discard(self)
      return self.op.src[0].permute_op(tuple([self.op.arg[i] for i in arg]))
    if not realized:
      if PUSH_PERMUTES and self.optype == ReduceOps:
        # reduceops have one buffer input, permute it
        narg = tuple([self.op.arg[arg[i]] for i in range(len(arg))])
        src, rop = self.op.src[0], self.op.op
        src.children.discard(self)
        del self  # TODO: why doesn't this delete remove it from the children
        return src.permute_op(arg).reduce_op(rop, narg)

      # move permutes before expands (always, this is safe)
      if self.op.op == MovementOps.EXPAND:
        self.op.src[0].children.discard(self)
        return self.op.src[0].permute_op(arg).expand_op(tuple([self.op.arg[a] for a in arg]))

      # move permutes before reshapes if we can
      if PUSH_PERMUTES and self.op.op == MovementOps.RESHAPE and self.op.src[0].__class__ == LazyBuffer:
        if shape_idx_groups := get_contraction(self.op.src[0].shape, shape):
          self.op.src[0].children.discard(self)   # this changes nothing?
          return self.op.src[0].permute_op(tuple(flatten(shape_idx_groups[i] for i in arg))) \
            .reshape_op(ShapeTracker(self.st).permute(arg).shape)
      if SHUFFLE_MOVEMENT_OPS and self.optype == BinaryOps and not realized and len(self.children) == 0:
        return self.op.replace_with_movement_ops([(MovementOps.PERMUTE, arg)])
    ret = create_lazybuffer(self.device, ShapeTracker(self.st).permute(arg), MovementOps, LazyOp(MovementOps.PERMUTE, (self,), arg), self.dtype)
    if REMOVE_MOVEMENT_NOPS and not ret.realized and not realized and ret.st.contiguous:
      # MovementOps aren't stacked any more, they each have one parent, find the root
      root = get_movementroot(self)
      if root.st.contiguous and root != self and prod(ret.st.shape) == prod(root.shape):
        return root.reshape_op(ret.st.shape)

    return ret
  
  def shrink_op(self:LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer:
    shape, realized = self.shape, self.realized
    if all([b - a == s for s, (a, b) in zip(shape, arg)]): return self
    if not realized and self.op.op == MovementOps.SHRINK:
      self.op.src[0].children.discard(self)
      return self.op.src[0].shrink_op(tuple([(b1+b2, b1+e2) for (b1,e1),(b2,e2) in zip(self.op.arg, arg)]))
    if SHUFFLE_MOVEMENT_OPS and self.optype == BinaryOps and not realized and len(self.children) == 0:
      return self.op.replace_with_movement_ops([(MovementOps.SHRINK, arg)])
    ret = create_lazybuffer(self.device, ShapeTracker(self.st).shrink(arg), MovementOps, LazyOp(MovementOps.SHRINK, (self,), arg), self.dtype)
    if REMOVE_MOVEMENT_NOPS and not ret.realized and not realized and ret.st.contiguous:
      # MovementOps aren't stacked any more, they each have one parent, find the root
      root = get_movementroot(self)
      if root.st.contiguous and root != self and prod(ret.st.shape) == prod(root.shape):
        return root.reshape_op(ret.st.shape)
    return ret
  
  def stride_op(self:LazyBuffer, arg:Tuple[int, ...]) -> LazyBuffer:
    shape, realized = self.shape, self.realized
    local_st = ShapeTracker(shape).stride(arg)
    if shape == local_st.shape and local_st.contiguous: return self
    if not realized and self.op.op == MovementOps.STRIDE:
      self.op.src[0].children.discard(self)
      return self.op.src[0].stride_op(tuple([i*j for i,j in zip(arg, self.op.arg)]))
    if SHUFFLE_MOVEMENT_OPS and self.optype == BinaryOps and not realized and len(self.children) == 0:
      return self.op.replace_with_movement_ops([(MovementOps.STRIDE, arg)])
    ret = create_lazybuffer(self.device, ShapeTracker(self.st).stride(arg), MovementOps, LazyOp(MovementOps.STRIDE, (self,), arg), self.dtype)
    if REMOVE_MOVEMENT_NOPS and not ret.realized and not realized and ret.st.contiguous:
      # MovementOps aren't stacked any more, they each have one parent, find the root
      root = get_movementroot(self)
      if root.st.contiguous and root != self and prod(ret.st.shape) == prod(root.shape):
        return root.reshape_op(ret.st.shape)
    return ret

  def map_buffers(self, real_srcs: Dict[Any, Any]):
    if self in real_srcs:
      return real_srcs[self]
    else:
      return self
  
  def get_buffers(self) -> Tuple[LazyBuffer]: return (self,)
  def get_lazyops(self) -> List[Any]: return []
  def replace_with_movement_ops(self: LazyBuffer, ops:List[Tuple[MovementOps, Tuple[Any, ...]]]) -> LazyBuffer:
    y = self
    for op, arg in ops: y = MOVEMENT_OPS_DISPATCHER[op](y, arg)
    return y
  
def _push_movement_ops(srcs:Tuple[LazyBuffer, ...]) -> Tuple[LazyBuffer, ...]:
  new_srcs = []
  for x in srcs:
    mops: List[Tuple[MovementOps, Tuple[Any, ...]]] = []
    bx = x
    # backwalk all the movement ops. don't push PAD or EXPAND
    while not bx.realized and bx.optype == MovementOps and bx.op.op != MovementOps.EXPAND and (SHUFFLE_PAD_OPS or bx.op.op != MovementOps.PAD) and len(bx.children) <= 1:
      assert bx.op.op in MOVEMENT_OPS
      mops.append((bx.op.op, bx.op.arg))
      bx = bx.op.src[0]
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
    new_srcs = []
    for x in srcs:
      if not x.realized and x.op.op == LoadOps.CONTIGUOUS and len(x.op.src[0].children) <= 1:
        x.op.src[0].children.discard(x)
        new_srcs.append(x.op.src[0])
      else:
        new_srcs.append(x)
    return elementwise_op(op, *new_srcs, arg=arg).contiguous()

  if MERGE_ELEMENTWISE_OPS:
    # remove the buffers from any (childless) BinaryOps that feed into this
    srcs = tuple([x.op if x.optype == BinaryOps and len(x.children) == 0 and not x.realized else x for x in srcs])  # type: ignore

  return create_lazybuffer(out_device, out_shape, BinaryOps, LazyOp(op, srcs, arg), out_dtype)

class _Device:
  def __init__(self) -> None:
    self._buffers: List[str] = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
    self.DEFAULT: str = functools.reduce(lambda val, ele: ele if getenv(ele) == 1 else val, self._buffers, self._default_device())
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

MOVEMENT_OPS_DISPATCHER = {
  MovementOps.RESHAPE: LazyBuffer.reshape_op,
  MovementOps.EXPAND: LazyBuffer.expand_op,
  MovementOps.SHRINK: LazyBuffer.shrink_op,
  MovementOps.PERMUTE: LazyBuffer.permute_op,
  MovementOps.PAD: LazyBuffer.pad_op,
  MovementOps.STRIDE: LazyBuffer.stride_op,
}

def _realize_fromcpu(buffer: LazyBuffer) -> None:
  if prod(buffer.op.arg.shape) == 1 and hasattr(Device[buffer.device].codegen, 'supports_constant_folding'):
    buffer.realized = RawConst(1, dtypes.from_np(buffer.op.arg.dtype), buffer.op.arg().flatten()[0])
  else:
    if DEBUG >= 4: print(f"copying {buffer.op.arg.shape}:{dtypes.from_np(buffer.op.arg.dtype)} -> {buffer.device}")
    buffer.realized = Device[buffer.device].buffer.fromCPU(buffer.op.arg(), **buffer._device_extra_args())


def _realize_contiguous(buffer: LazyBuffer) -> None:
  realized = buffer.op.src[0].realize().realized
  if buffer.op.src[0].st.contiguous and not realized.__class__ == RawConst and realized.size == prod(buffer.shape):
    # no need to run an AST, this is already contiguous
    buffer.realized = realized
  else:
    # TODO: remove UnaryOps.NOOP, replace with LoadOps.CONTIGUOUS. confusing with Compiled though
    buffer.op = LazyOp(UnaryOps.NOOP, buffer.op.src)

def _realize_custom(buffer: LazyBuffer) -> None:
  # this needs to immediately realize
  buffer.realized = buffer.op.arg(buffer, *[x.realize() for x in buffer.op.src])

def _realize_empty(buffer: LazyBuffer) -> None:
  buffer.realized = Device[buffer.device].buffer(prod(buffer.shape), buffer.dtype, **buffer._device_extra_args())

REALIZE_DISPATCHER = {
  LoadOps.FROMCPU: _realize_fromcpu,
  LoadOps.CONTIGUOUS: _realize_contiguous,
  LoadOps.CUSTOM: _realize_custom,
  LoadOps.EMPTY: _realize_empty,
}