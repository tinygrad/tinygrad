from __future__ import annotations
from typing import Optional, Tuple, Union, List, Dict, Any, cast
import sys, weakref, importlib, inspect, functools, pathlib
from weakref import WeakValueDictionary
from tinygrad.helpers import prod, getenv, DType, dtypes, LazyNumpyArray, flatten, ImageDType
from tinygrad.shape.shapetracker import ShapeTracker, get_contraction
from tinygrad.ops import Compiled, Interpreted, UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, OpType, LazyOp, get_lazyops, get_buffers, map_buffers
from tinygrad.runtime.lib import RawConst, RawBuffer

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
  if MERGE_ELEMENTWISE_INTO_REDUCE and src.realized is None and src.optype == BinaryOps and len(src.children) <= 1:
    src = src.op
  return LazyOp(self.op.op, (src,), self.op.arg)

# this supports late merging an upstream Reduce op and even an Elementwise op above that
def _ast_binaryops(self:LazyBuffer) -> LazyOp:
  real_srcs: Dict[LazyBuffer, Union[None, LazyOp, LazyBuffer]] = {x:None for x in get_buffers(self.op)}
  # NOTE: contiguous does not always mean the same size with SHRINK. this is still mergeable but requires more thought how
  # TODO: this can also support late fusion of BinaryOps, required for test_fold_conv_sgd
  psrcs: List[Tuple[LazyBuffer, LazyBuffer]] = [(k,x) for k,x in zip(real_srcs.keys(), map(get_movementroot_contiguous, real_srcs.keys())) if x.optype == ReduceOps and x.realized is None and prod(k.shape) == prod(x.shape) and len(x.children) <= 1 and len(k.children) <= 1]
  intermediate_shape: Tuple[int, ...] = self.shape
  if len(psrcs) == 1 and MERGE_ONE_REDUCE_INTO_ELEMENTWISE:
    psrc = psrcs[0] # NOTE: right now we can't handle multiple, as we'd have to check for loop
    if psrc[1].optype == ReduceOps:
      top = _ast_reduceops(psrc[1])
    real_srcs[psrc[0]] = top
    real_srcs.update({x:x for x in get_buffers(top)})  # the reduce op buffers are not modified

    # if the ReduceOp is followed by a reshape, we push this reshape before all the ElementwiseOp inputs
    if psrc[0].shape != psrc[1].shape:
      intermediate_shape = psrc[1].shape
      assert psrc[0].shape == self.shape, f"shape mismatch {psrc[0].shape} != {self.shape}"

  # reshape all the late ops into the output shape
  # NOTE: these RESHAPEs will return self if they don't change the shape
  for x in real_srcs.keys():
    if real_srcs[x] is None: real_srcs[x] = x.movement_op(MovementOps.RESHAPE, intermediate_shape)
  ast = map_buffers(real_srcs, self.op)
  return LazyOp(MovementOps.RESHAPE, (ast, ), self.shape) if intermediate_shape != self.shape else ast

# **** lazy operations ****

def get_weakop(op:LazyOp) -> LazyOp: return LazyOp(op.op, tuple(get_weakop(x) if isinstance(x, LazyOp) else weakref.ref(x) for x in op.src), op.arg)
def get_single_root(root:LazyBuffer) -> LazyBuffer: return get_single_root(root.op.src[0]) if getattr(root, 'op', None) and len(root.op.src) == 1 else root
def get_movementroot(root:LazyBuffer, allow_contiguous=False) -> LazyBuffer: return get_movementroot(root.op.src[0], allow_contiguous) if root.realized is None and (root.optype == MovementOps or (root.op.op == LoadOps.CONTIGUOUS and allow_contiguous and root.op.src[0].st.contiguous)) else root
def get_movementroot_contiguous(x:LazyBuffer) -> LazyBuffer: return get_movementroot_contiguous(x.op.src[0]) if x.realized is None and x.op.op == LoadOps.CONTIGUOUS else (get_movementroot(x, True) if x.optype == MovementOps and x.st.contiguous else x)

def replace_with_movement_ops(y:Union[LazyOp, LazyBuffer], ops:List[Tuple[MovementOps, Tuple[Any, ...]]]) -> LazyBuffer:
  if isinstance(y, LazyBuffer):
    for op, arg in ops: y = y.movement_op(op, arg)
    return y
  assert y.op in BinaryOps or y.op in UnaryOps
  return elementwise_op(y.op, *[replace_with_movement_ops(z, ops) for z in y.src], arg=y.arg)   # type: ignore

lazycache: WeakValueDictionary[Tuple[str, DType, OpType, LazyOp], LazyBuffer] = WeakValueDictionary()
def create_lazybuffer(device:str, shape:Union[ShapeTracker, Tuple[int, ...]], optype:OpType, op:LazyOp, dtype:DType):
  st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))

  # fromcpu aren't cached
  if optype == LoadOps and op.op == LoadOps.FROMCPU: return LazyBuffer(device, st, optype, op, dtype)

  #print("create_lazybuffer", device, shape, optype, op, dtype)

  # NOTE: shape should be deterministic. annoying to cache with the ShapeTracker
  # get_weakop makes all the LazyBuffers in the op have a weakref
  wop = (device, dtype, optype, get_weakop(op))

  if wop not in lazycache: lazycache[wop] = ret = LazyBuffer(device, st, optype, op, dtype)
  else: ret = lazycache[wop]
  return ret

class LazyBuffer:
  __deletable__ = ('op',)
  def __init__(self, device:str, st:ShapeTracker, optype:OpType, op:LazyOp, dtype:DType):
    self.st = st  # NOTE: this is not a copy! this should be a "read-only" ShapeTracker
    self.device, self.shape, self.optype, self.dtype = device, self.st.shape, optype, dtype
    self.op: LazyOp = op
    self.realized: Optional[RawBuffer] = None
    self.output_buffer: Optional[RawBuffer] = None   # TODO: do we really need this? or can we just use realized
    # TODO: does children have to be a ref count instead of a set? can a Buffer be a double child?
    self.children: weakref.WeakSet[LazyBuffer] = weakref.WeakSet()
    # NOTE: op should be read only after construction of LazyBuffer
    for x in get_buffers(op): x.children.add(self)
    if not LAZY: self.realize()

    # log phantom ops to the graph
    from tinygrad.graph import log_op, GRAPH
    if GRAPH >= 3: log_op(self, self.op, phantom=True)

  def __repr__(self): return f"<LB {self.shape} {self.dtype} op:{self.op.op if self.realized is None else self.realized} st:{self.st}>"

  def realize(self:LazyBuffer) -> LazyBuffer:
    if self.realized is None:
      # get real ops first
      if self.op.op == LoadOps.FROMCPU:
        if prod(self.op.arg.shape) == 1 and hasattr(Device[self.device].codegen, 'supports_constant_folding'):
          self.realized = RawConst(1, dtypes.from_np(self.op.arg.dtype), self.op.arg().flatten()[0])
        else:
          self.realized = Device[self.device].buffer.fromCPU(self.op.arg())
      elif self.op.op == LoadOps.CONTIGUOUS:
        realized = self.op.src[0].realize().realized
        if self.op.src[0].st.contiguous and not isinstance(realized, RawConst) and realized.size == prod(self.shape):
          # no need to run an AST, this is already contiguous
          self.realized = realized
        else:
          # TODO: remove UnaryOps.NOOP, replace with LoadOps.CONTIGUOUS. confusing with Compiled though
          self.op = LazyOp(UnaryOps.NOOP, self.op.src)
      elif self.op.op == LoadOps.CUSTOM:
        # this needs to immediately realize
        self.realized = self.op.arg(self, *[x.realize() for x in self.op.src])
      # these can be late folded and change the op to go further back in the graph
      elif self.optype == ReduceOps: self.op = _ast_reduceops(self)
      elif self.optype == BinaryOps: self.op = _ast_binaryops(self)  # ISSUE: this can include a reshape

      # run the ast if we still have to, and log the op
      if self.realized is None:
        for x in get_buffers(self.op): x.realize()

        # HACK: image shape can be wrong, hot cast it back to a normal float
        if self.optype != MovementOps and isinstance(self.dtype, ImageDType) and (prod(self.shape) != prod(self.dtype.shape) or not any(self.shape[x]%4 == 0 for x in self.st.unit_stride_axes())):
          if self.op.op == MovementOps.RESHAPE:
            # put CAST before the final RESHAPE
            self.op = LazyOp(MovementOps.RESHAPE, (LazyOp(UnaryOps.CAST, self.op.src, dtypes.float32),), self.op.arg)
          else:
            self.op = LazyOp(UnaryOps.CAST, (self.op,), dtypes.float32)
          self.dtype = dtypes.float32

        self.realized = Device[self.device].exec_ast(self.op, output=self)

      assert isinstance(self.realized, (RawConst, Device[self.device].buffer)), f"device mismatch on realized got {type(self.realized)} expected {self.device}"
      # HACK: allow hot casting of images
      assert self.realized.dtype == self.dtype or self.dtype.name.startswith("image"), f"dtype mismatch on realize got {self.realized.dtype} expected {self.dtype}"
      self.dtype = self.realized.dtype

      # log to the graph
      from tinygrad.graph import log_op, GRAPH
      if not isinstance(self.realized, RawConst) or GRAPH >= 2: log_op(self, self.op)

      # no need to keep the op after realization
      del self.op
    return self

  # NOTE: we have to make a copy of the numpy array here in case the user changes it. expose this? LazyNumpyArray doesn't have this problem
  @staticmethod
  def fromCPU(x:LazyNumpyArray, device) -> LazyBuffer:
    return create_lazybuffer(device, x.shape, LoadOps, LazyOp(LoadOps.FROMCPU, tuple(), x.copy()), dtypes.from_np(x.dtype))

  # create a constant with the shape and dtype of self
  def const_like(self, val) -> LazyBuffer:
    return create_lazybuffer(self.device, (1,), LoadOps, LazyOp(LoadOps.FROMCPU, tuple(), LazyNumpyArray([val], (1,), self.dtype.np)), self.dtype) \
      .movement_op(MovementOps.RESHAPE, (1,)*len(self.shape)).movement_op(MovementOps.EXPAND, self.shape)

  # NOTE: we also have to copy the numpy array on the way out...otherwise the underlying Tensor could be freed and use after free. improve this?
  def toCPU(self):
    realized = self.cast(dtypes.from_np(self.dtype.np)).contiguous().realize().realized
    ret = cast(RawBuffer, realized).toCPU().reshape(self.shape)
    return ret.copy()

  def cast(self:LazyBuffer, arg:DType) -> LazyBuffer: return elementwise_op(UnaryOps.CAST, self, arg=arg) if self.dtype != arg else self
  def unary_op(self:LazyBuffer, op:UnaryOps) -> LazyBuffer: return elementwise_op(op, self)
  def binary_op(self:LazyBuffer, op:BinaryOps, y:LazyBuffer) -> LazyBuffer: return elementwise_op(op, self, y)
  def contiguous(self:LazyBuffer) -> LazyBuffer:
    if self.realized is None and self.op.op == LoadOps.CONTIGUOUS: return self  # two CONTIGUOUS in a row is one
    return create_lazybuffer(self.device, self.shape, LoadOps, LazyOp(LoadOps.CONTIGUOUS, (self,)), self.dtype)

  def reduce_op(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[int, ...]) -> LazyBuffer:
    if self.shape == tuple(new_shape): return self
    srcs = _push_movement_ops((self,)) if SHUFFLE_MOVEMENT_OPS else (self,)
    return create_lazybuffer(self.device, new_shape, ReduceOps, LazyOp(op, tuple(srcs), new_shape), self.dtype)

  # shrink -> stride -> permute -> reshape -> pad -> expand
  def movement_op(self:LazyBuffer, op:MovementOps, arg:Tuple[Any, ...]) -> LazyBuffer:
    # very instant nop
    if op == MovementOps.RESHAPE and self.shape == arg: return self

    # TODO: look into why that copy is needed
    local_st = ShapeTracker(self.shape).movement_op(op, arg)

    # instant nops
    if local_st.contiguous and self.shape == local_st.shape: return self

    # two ops in a row is one op. merge them if unresolved
    if self.realized is None and self.op.op == op:
      # TODO: why is deleting self from children needed? shouldn't GC do it?
      self.op.src[0].children.discard(self)
      if op in [MovementOps.RESHAPE, MovementOps.EXPAND]: return self.op.src[0].movement_op(op, arg)
      if op == MovementOps.SHRINK: return self.op.src[0].movement_op(op, tuple((b1+b2, b1+e2) for (b1,e1),(b2,e2) in zip(self.op.arg, arg)))
      if op == MovementOps.PERMUTE: return self.op.src[0].movement_op(op, tuple(self.op.arg[i] for i in arg))
      if op == MovementOps.PAD: return self.op.src[0].movement_op(op, tuple((b1+b2, e1+e2) for (b1,e1),(b2,e2) in zip(self.op.arg, arg)))
      if op == MovementOps.STRIDE: return self.op.src[0].movement_op(op, tuple(i*j for i,j in zip(arg, self.op.arg)))

    # push permutes before reduce ops
    if op == MovementOps.PERMUTE and PUSH_PERMUTES and self.realized is None and self.optype == ReduceOps:
      # reduceops have one buffer input, permute it
      narg = tuple(self.op.arg[arg[i]] for i in range(len(arg)))
      src, rop = self.op.src[0], self.op.op
      src.children.discard(self)
      del self  # TODO: why doesn't this delete remove it from the children
      return src.movement_op(op, arg).reduce_op(rop, narg)

    # some permutes are actually just reshapes
    if op == MovementOps.PERMUTE and local_st.contiguous: return self.movement_op(MovementOps.RESHAPE, tuple(self.shape[i] for i in arg))

    # move permutes before expands (always, this is safe)
    if op == MovementOps.PERMUTE and self.realized is None and self.op.op == MovementOps.EXPAND:
      self.op.src[0].children.discard(self)
      return self.op.src[0].movement_op(MovementOps.PERMUTE, arg).movement_op(MovementOps.EXPAND, tuple(self.op.arg[a] for a in arg))

    # move permutes before reshapes if we can
    if op == MovementOps.PERMUTE and PUSH_PERMUTES and self.realized is None and self.op.op == MovementOps.RESHAPE and isinstance(self.op.src[0], LazyBuffer):
      if shape_idx_groups := get_contraction(self.op.src[0].shape, self.shape):
        self.op.src[0].children.discard(self)   # this changes nothing?
        return self.op.src[0].movement_op(MovementOps.PERMUTE, tuple(flatten(shape_idx_groups[i] for i in arg))) \
          .movement_op(MovementOps.RESHAPE, ShapeTracker(self.st).movement_op(op, arg).shape)

    # if this MovementOp is being applied to a BinaryOp, apply the MovementOp to all the BinaryOp inputs instead. NOTE: UnaryOps is never an OpType
    if SHUFFLE_MOVEMENT_OPS and self.optype == BinaryOps and self.realized is None and (op in [MovementOps.SHRINK, MovementOps.STRIDE, MovementOps.PERMUTE] or (op == MovementOps.RESHAPE and self.op.op in UnaryOps)) and len(self.children) == 0: # and op != MovementOps.EXPAND and (op != MovementOps.PAD or (SHUFFLE_PAD_OPS and all(x.op != BinaryOps.DIV for x in get_lazyops(self.op)))):
      return replace_with_movement_ops(self.op, [(op, arg)])

    # create the buffer
    ret = create_lazybuffer(self.device, ShapeTracker(self.st).movement_op(op, arg), MovementOps, LazyOp(op, (self,), arg), self.dtype)

    # if the ShapeTracker becomes contiguous, replace the whole thing with a reshape (or nothing if shapes match)
    # NOTE: if ret is in the cache, it can already be realized
    if REMOVE_MOVEMENT_NOPS and ret.realized is None and self.realized is None and ret.st.contiguous:
      # MovementOps aren't stacked any more, they each have one parent, find the root
      root = get_movementroot(self)
      if root.st.contiguous and root != self and prod(ret.st.shape) == prod(root.shape):
        return root.movement_op(MovementOps.RESHAPE, ret.st.shape)

    return ret

def _push_movement_ops(srcs:Tuple[LazyBuffer, ...]) -> Tuple[LazyBuffer, ...]:
  new_srcs = []
  for x in srcs:
    mops: List[Tuple[MovementOps, Tuple[Any, ...]]] = []
    bx = x
    # backwalk all the movement ops. don't push PAD or EXPAND
    while bx.realized is None and bx.optype == MovementOps and bx.op.op != MovementOps.EXPAND and (bx.op.op != MovementOps.PAD or SHUFFLE_PAD_OPS) and len(bx.children) <= 1:
      assert isinstance(bx.op.op, MovementOps)
      mops.append((bx.op.op, bx.op.arg))
      bx = bx.op.src[0]
    # NOTE: can't push pads with a div
    if bx.realized is None and bx.optype == BinaryOps and len(bx.children) <= 1 and len(mops) and (all(x[0] != MovementOps.PAD for x in mops) or all(x.op != BinaryOps.DIV for x in get_lazyops(bx.op))):
      new_srcs.append(replace_with_movement_ops(bx.op, mops[::-1]))
    else:
      new_srcs.append(x)
  return tuple(new_srcs)

def elementwise_op(op:Union[UnaryOps, BinaryOps], *srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
  # if we are separated from other binary ops by movement ops, we push those movement ops above those binaryops
  if SHUFFLE_MOVEMENT_OPS: srcs = _push_movement_ops(srcs)

  # get outputs now
  out_device, out_shape, out_dtype = srcs[0].device, srcs[0].shape, max(x.dtype for x in srcs) if op != UnaryOps.CAST else cast(DType, arg)

  # push all contiguous to the end of BinaryOps. kernels 198 -> 196
  if PUSH_CONTIGUOUS and any(x.realized is None and x.op.op == LoadOps.CONTIGUOUS and len(x.op.src[0].children) <= 1 for x in srcs):
    new_srcs = []
    for x in srcs:
      if x.realized is None and x.op.op == LoadOps.CONTIGUOUS and len(x.op.src[0].children) <= 1:
        x.op.src[0].children.discard(x)
        new_srcs.append(x.op.src[0])
      else:
        new_srcs.append(x)
    return elementwise_op(op, *new_srcs, arg=arg).contiguous()

  if MERGE_ELEMENTWISE_OPS:
    # remove the buffers from any (childless) BinaryOps that feed into this
    srcs = tuple(x.op if x.optype == BinaryOps and len(x.children) == 0 and x.realized is None else x for x in srcs)  # type: ignore

  return create_lazybuffer(out_device, out_shape, BinaryOps, LazyOp(op, srcs, arg), out_dtype)

class _Device:
  def __init__(self) -> None:
    self._buffers: List[str] = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
    self.DEFAULT: str = functools.reduce(lambda val, ele: ele if getenv(ele) == 1 else val, self._buffers, self._default_device())
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __getitem__(self, x:str) -> Union[Interpreted, Compiled]: return [cls for cname, cls in inspect.getmembers(importlib.import_module(f'tinygrad.runtime.ops_{x.lower()}')) if (cname.lower() == x.lower() + "buffer") and x in self._buffers][0]
  def _default_device(self) -> str:
    for device in ["METAL", "CUDA", "GPU"]:
      try:
        if self[device]: return device
      except Exception: pass
    return "CPU"
Device = _Device()
