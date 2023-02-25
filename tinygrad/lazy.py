from __future__ import annotations
from typing import Optional, Tuple, Union, List, Dict, Any, ClassVar, Type
import sys, weakref, importlib, inspect
from weakref import WeakValueDictionary
from tinygrad.helpers import ConvArgs, prod, DEBUG
from tinygrad.shape import ShapeTracker
from tinygrad.ops import DeviceBuffer, UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps, OpType, LazyOp, get_buffers, get_lazyops, map_buffers, GenericExecAST
from tinygrad.graph import log_op
from tinygrad.helpers import getenv

# lazy can recurse a lot
sys.setrecursionlimit(10000)

OPT = getenv("OPT", 2)
NOCONV = getenv("NOCONV", 0)
IMAGE = getenv("IMAGE", 0)
LAZY = getenv("LAZY", 1)

def get_buffer(name, base='tinygrad.llops'):
  try:
    return (name.upper(), [cls for cname, cls in inspect.getmembers(importlib.import_module(f'{base}.ops_{name}'), inspect.isclass) if (cname.lower() == name + "buffer")][0])
  except Exception as e:  # NOTE: this can't be put on one line due to mypy issue
    print(name, "backend not available", e, file=sys.stderr)

class _Device:
  def __init__(self) -> None:
    self._buffers : Dict[str, Type[DeviceBuffer]] = {x[0]:x[1] for x in [
      get_buffer('cpu'), get_buffer('gpu'), get_buffer('llvm'), get_buffer('torch'),
      get_buffer('triton', 'accel.triton')] if x is not None}
    self.DEFAULT : str = "CPU"
    for name in self._buffers:
      if getenv(name) == 1: self.DEFAULT = name  # note: DEFAULT can be a Device that can't be imported. better than silent use of a different device
      self.__setattr__(name, name)
Device = _Device()

# TODO: movement ops that only change shape are really nops. treat them as such
REMOVE_MOVEMENT_NOPS, MERGE_UNARY_OPS, MERGE_ELEMENTWISE_INTO_REDUCE, SHUFFLE_MOVEMENT_OPS = OPT>=1, OPT>=1, OPT>=1, OPT>=1
MERGE_ELEMENTWISE_OPS, MERGE_ONE_REDUCE_INTO_ELEMENTWISE = OPT>=2, OPT>=2
PUSH_PERMUTES = OPT>=3    # fairly untested, but gets kernels back to 200 for openpilot

# **** realize functions ****
def _ast_reduceops(self:LazyBuffer) -> LazyOp:
  # TODO: this can also corealize a binary op after the reduce, not just before
  src = self.op.src[0]
  if MERGE_ELEMENTWISE_INTO_REDUCE and src.realized is None and src.optype == BinaryOps and len(src.children) <= 1:
    src = src.op
  return LazyOp(self.op.op, (src,), self.op.arg)

# this supports late merging an upstream Reduce op and even an Elementwise op above that
def _ast_binaryops(self:LazyBuffer) -> LazyOp:
  real_srcs : Dict[LazyBuffer, Union[None, LazyOp, LazyBuffer]] = {x:None for x in get_buffers(self.op)}
  if DEBUG >= 3:
    for k,x in zip(real_srcs.keys(), map(get_movementroot_contiguous, real_srcs.keys())):
      if x.optype in [ProcessingOps,ReduceOps] and x.realized is None:
        print("\nHIT", k,x)
        for tk in k.children: print("k", tk)
        for tx in x.children: print("x", tx)
  # NOTE: contiguous does not always mean the same size with SHRINK. this is still mergeable but requires more thought how
  psrcs : List[Tuple[LazyBuffer, LazyBuffer]] = [(k,x) for k,x in zip(real_srcs.keys(), map(get_movementroot_contiguous, real_srcs.keys())) if x.optype in [ProcessingOps,ReduceOps] and x.realized is None and prod(k.shape) == prod(x.shape) and len(x.children) <= 1 and len(k.children) <= 1]
  intermediate_shape : Tuple[int, ...] = self.shape
  if len(psrcs) == 1 and MERGE_ONE_REDUCE_INTO_ELEMENTWISE:
    if psrcs[0][1].optype == ProcessingOps:
      top = psrcs[0][1].op  # _ast_processingops
    elif psrcs[0][1].optype == ReduceOps:
      top = _ast_reduceops(psrcs[0][1])
    real_srcs[psrcs[0][0]] = top
    real_srcs.update({x:x for x in get_buffers(top)})  # the reduce op buffers are not modified

    # if the ReduceOp is followed by a reshape, we push this reshape before all the ElementwiseOp inputs
    if psrcs[0][0].shape != psrcs[0][1].shape:
      intermediate_shape = psrcs[0][1].shape
      assert psrcs[0][0].shape == self.shape, f"shape mismatch {psrcs[0][0].shape} != {self.shape}"

  # reshape all the late ops into the output shape
  # NOTE: these RESHAPEs will return self if they don't change the shape
  for x in real_srcs.keys():
    if real_srcs[x] is None:
      real_srcs[x] = x.movement_op(MovementOps.RESHAPE, intermediate_shape)
  ast = map_buffers(real_srcs, self.op)
  return LazyOp(MovementOps.RESHAPE, (ast, ), self.shape) if intermediate_shape != self.shape else ast

# **** lazy operations ****

def get_weakop(op:LazyOp) -> LazyOp: return LazyOp(op.op, tuple(get_weakop(x) if isinstance(x, LazyOp) else weakref.ref(x) for x in op.src), op.arg)
def get_single_root(root:LazyBuffer) -> LazyBuffer: return get_single_root(root.op.src[0]) if getattr(root, 'op', None) and len(root.op.src) == 1 else root
def get_movementroot(root:LazyBuffer) -> LazyBuffer: return get_movementroot(root.op.src[0]) if root.realized is None and (root.optype == MovementOps or (root.op.op == LoadOps.CONTIGUOUS and root.op.src[0].st.contiguous)) else root
def get_movementroot_contiguous(x:LazyBuffer) -> LazyBuffer: return get_movementroot(x) if x.optype == MovementOps and x.st.contiguous else x

def replace_with_movement_op(y:Union[LazyOp, LazyBuffer], op:MovementOps, arg:Tuple[Any, ...]) -> LazyBuffer:
  if isinstance(y, LazyBuffer): return y.movement_op(op, arg)
  assert y.op in BinaryOps or y.op in UnaryOps
  return elementwise_op(y.op, *[replace_with_movement_op(z, op, arg) for z in y.src])   # type: ignore

def support_weakref(x): return x
@support_weakref  # needed for mypyc, this prevents LazyBuffer from becoming a native class
class LazyBuffer:
  __deletable__ = ('op',)
  lazycache : ClassVar[WeakValueDictionary[Tuple[str, OpType, LazyOp], LazyBuffer]] = WeakValueDictionary()
  def __new__(cls, device:str, shape:Union[ShapeTracker, Tuple[int, ...]], optype:OpType, op:LazyOp):
    # fromcpu aren't cached
    if optype == LoadOps and op.op == LoadOps.FROMCPU:
      return super().__new__(cls)
    wop = (device, optype, get_weakop(op))   # NOTE: shape should be deterministic. annoying to cache with the ShapeTracker
    # NOTE: we need "ret" to prevent the new buffer from being immediately deleted
    if wop not in LazyBuffer.lazycache: LazyBuffer.lazycache[wop] = ret = super().__new__(cls)
    else: ret = LazyBuffer.lazycache[wop]
    return ret

  def __init__(self, device:str, shape:Union[ShapeTracker, Tuple[int, ...]], optype:OpType, op:LazyOp):
    if hasattr(self, 'device'):
      return  # cache hit, we return and don't reinit
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape, self.optype, self.op = self.st.shape, optype, op
    self.realized : Optional[DeviceBuffer] = None
    self.output_buffer : Optional[DeviceBuffer] = None
    self.device, self.dbuffer = device, Device._buffers[device]
    self.children : weakref.WeakSet[LazyBuffer] = weakref.WeakSet()
    # NOTE: op should be read only after construction of LazyBuffer
    for x in get_buffers(op):
      x.children.add(self)
    if not LAZY:
      self.realize()
    if DEBUG >= 4: print(f"create {self}")

  def __repr__(self): return f"<LB {self.shape} op:{self.op.op if self.realized is None else 'realized'}>"

  # this produces a device buffer
  def realize(self:LazyBuffer, required_device=None) -> DeviceBuffer:
    if required_device is not None:
      assert required_device == self.device
    if self.realized is None:
      # get real ops first
      if self.op.op == LoadOps.FROMCPU:
        self.realized = Device._buffers[self.device].fromCPU(self.op.arg)
        ast = LazyOp(self.op.op, tuple())
      elif self.op.op == LoadOps.CONTIGUOUS:
        real_src = self.op.src[0].realize(self.device)
        self.realized = real_src.contiguous()
        ast = LazyOp(self.op.op, (real_src, ))
      elif self.optype == MovementOps:
        src = self.op.src[0]

        # fuse RESHAPE and ReduceOps
        if src.realized is None and src.optype == ReduceOps and self.op.op == MovementOps.RESHAPE and len(src.children) <= 1:
          # it's okay to add a RESHAPE to the ast here
          ast = LazyOp(MovementOps.RESHAPE, (_ast_reduceops(src), ), self.op.arg)
        else:
          # movement ops aren't an AST, just run them
          real_src = src.realize(self.device)
          self.realized = real_src.movement_op(self.op.op, self.op.arg)
          ast = LazyOp(self.op.op, (real_src, ))
      elif self.optype == ProcessingOps: ast = self.op   # no ast modifications for ProcessingOps
      elif self.optype == ReduceOps: ast = _ast_reduceops(self)
      elif self.optype == BinaryOps: ast = _ast_binaryops(self)

      # no need to keep the op after realization
      del self.op

      # run the ast if we still have to, and log the op
      if self.realized is None:
        ast = map_buffers({x:x.realize(self.device) for x in get_buffers(ast)}, ast)
        self.realized = self.dbuffer.exec_ast(ast, output_buffer=self.output_buffer)
      log_op(self.realized, ast)

    assert self.realized.shape == self.shape, f"shape mismatch on realize {self.realized.shape} vs {self.shape}"
    assert isinstance(self.realized, Device._buffers[self.device])
    return self.realized

  @staticmethod
  def fromCPU(x, device): return LazyBuffer(device, x.shape, LoadOps, LazyOp(LoadOps.FROMCPU, tuple(), x.copy()))
  def toCPU(self): return self.realize().toCPU()

  def unary_op(self:LazyBuffer, op:UnaryOps) -> LazyBuffer: return elementwise_op(op, self)
  def binary_op(self:LazyBuffer, op:BinaryOps, y:LazyBuffer) -> LazyBuffer: return elementwise_op(op, self, y)
  def contiguous(self:LazyBuffer) -> LazyBuffer: return LazyBuffer(self.device, self.shape, LoadOps, LazyOp(LoadOps.CONTIGUOUS, (self,)))

  def reduce_op(self:LazyBuffer, op:ReduceOps, new_shape:Tuple[int, ...]) -> LazyBuffer:
    if self.shape == tuple(new_shape):
      return self
    reduce = list(enumerate(zip(self.shape, new_shape)))
    # move the reduce axes to the end
    x = self.movement_op(MovementOps.PERMUTE, tuple([i for i,(s,n) in reduce if s == n] + [i for i,(s,n) in reduce if s != n]))
    new_tmp_shape = tuple([n for _,(s,n) in reduce if s == n] + [n for _,(s,n) in reduce if s != n])
    # NOTE: this reshape can only move around 1s
    return LazyBuffer(x.device, new_tmp_shape, ReduceOps, LazyOp(op, (x,), new_tmp_shape)).movement_op(MovementOps.RESHAPE, new_shape)

  # syntactic sugar around PAD and SHRINK
  # TODO: turn RESHAPE into EXPAND and CONTRACT (current EXPAND should be REPEAT)
  def slice(self:LazyBuffer, arg):
    padding = tuple((max(0, -p[0]), max(0, p[1]-self.shape[i])) for i,p in enumerate(arg))
    return self.movement_op(MovementOps.PAD, padding).movement_op(MovementOps.SHRINK, tuple((p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg)))

  def movement_op(self:LazyBuffer, op:MovementOps, arg:Tuple[Any, ...]) -> LazyBuffer:
    # very instant nop
    if op == MovementOps.RESHAPE and self.shape == arg: return self

    # TODO: look into why that copy is needed
    local_st = ShapeTracker(self.shape).movement_op(op, arg)

    # instant nops
    if local_st.contiguous and self.shape == local_st.shape and op != MovementOps.STRIDED:
      return self

    # two ops in a row is one op. merge them if unresolved
    if self.realized is None and self.op.op == op:
      if op in [MovementOps.RESHAPE, MovementOps.EXPAND, MovementOps.SHRINK]:
        return self.op.src[0].movement_op(op, arg)
      if op == MovementOps.PERMUTE:
        return self.op.src[0].movement_op(op, tuple(self.op.arg[i] for i in arg))
      if op == MovementOps.PAD:
        return self.op.src[0].movement_op(op, tuple((b1+b2, e1+e2) for (b1,e1),(b2,e2) in zip(self.op.arg, arg)))
      # TODO: MovementOps.FLIP / MovementOps.STRIDED?

    # push permutes before reduce ops
    if op == MovementOps.PERMUTE and PUSH_PERMUTES and self.realized is None and self.optype == ReduceOps:
      # reduceops have one buffer input, permute it
      narg = tuple(self.op.arg[arg[i]] for i in range(len(arg)))
      src, rop = self.op.src[0], self.op.op
      src.children = [y for y in src.children if self != y]
      del self  # TODO: why doesn't this delete remove it from the children
      return src.movement_op(op, arg).reduce_op(rop, narg)

    # some permutes are actually just reshapes
    if op == MovementOps.PERMUTE and local_st.contiguous:
      return self.movement_op(MovementOps.RESHAPE, tuple(self.shape[i] for i in arg))

    # move permutes before reshapes if we can
    if op == MovementOps.PERMUTE and PUSH_PERMUTES and self.realized is None and self.op.op == MovementOps.RESHAPE and isinstance(self.op.src[0], LazyBuffer):
      # is contract? if so, group the axis
      def get_contraction(old_shape:Tuple[int, ...], new_shape:Tuple[int, ...]):
        out : List[List[int]] = []
        curr : List[int] = []
        for t in old_shape:
          if len(out) >= len(new_shape): break
          if t*prod(curr) <= new_shape[len(out)]:
            curr.append(t)
          else:
            out.append(curr)
            curr = [t]
        out.append(curr)
        if len(new_shape) == len(out) and all(prod(i) == j and len(i) >= 1 for i,j in zip(out, new_shape)):
          return out
      contraction = get_contraction(self.op.src[0].shape, self.shape)
      if contraction is not None:
        numbered = []
        start = 0
        for c in contraction:
          numbered.append(list(range(start, start+len(c))))
          start += len(c)
        new_arg = []
        for p in arg:
          new_arg += numbered[p]
        return self.op.src[0].movement_op(MovementOps.PERMUTE, tuple(new_arg)) \
          .movement_op(MovementOps.RESHAPE, ShapeTracker(self.st).movement_op(op, arg).shape)

    # some strideds are actually just reshapes
    # NOTE: due to how strided works, we have to check the parent to be contiguous also
    if op == MovementOps.STRIDED and local_st.contiguous and self.st.contiguous:
      return self.movement_op(MovementOps.RESHAPE, tuple(i for i,_ in arg))

    # if this MovementOp is being applied to a BinaryOp, apply the MovementOp to all the BinaryOp inputs instead. NOTE: UnaryOps is never an OpType
    if SHUFFLE_MOVEMENT_OPS and self.optype == BinaryOps and self.realized is None and len(self.children) == 0 and op not in [MovementOps.EXPAND, MovementOps.STRIDED] and (op != MovementOps.PAD or all(x.op != BinaryOps.DIV for x in get_lazyops(self.op))):
      return replace_with_movement_op(self.op, op, arg)

    # create the buffer
    ret = LazyBuffer(self.device, ShapeTracker(self.st).movement_op(op, arg), MovementOps, LazyOp(op, (self,), arg))

    # if the ShapeTracker becomes contiguous, replace the whole thing with a reshape (or nothing if shapes match)
    # NOTE: if ret is in the cache, it can already be realized
    if REMOVE_MOVEMENT_NOPS and ret.realized is None and self.realized is None and ret.st.contiguous:
      # MovementOps aren't stacked any more, they each have one parent, find the root
      root = get_movementroot(self)
      if root.st.contiguous and root != self and prod(ret.st.shape) == prod(root.shape):
        return root.movement_op(MovementOps.RESHAPE, ret.st.shape) if ret.st.shape != root.shape else root

    return ret

  def processing_op(self:LazyBuffer, op:ProcessingOps, w:LazyBuffer, C:ConvArgs) -> LazyBuffer:
    x = self

    if IMAGE >= 1:
      w = w.movement_op(MovementOps.RESHAPE, (C.groups, C.rcout, C.cin, C.H, C.W))
      # TODO: moving the x reshape here creates more views?

      # hack for non multiples of 4 on C.cin
      if C.cin % 4 != 0 and not (C.cin == 1 and C.groups%4 == 0):
        added_input_channels = 4 - (C.cin % 4)
        w = w.movement_op(MovementOps.PAD, tuple((0, added_input_channels) if i == 2 else (0, 0) for i in range(len(w.shape))))
        x = x.movement_op(MovementOps.RESHAPE, (C.bs, C.groups, C.cin, C.iy, C.ix))
        x = x.movement_op(MovementOps.PAD, tuple((0, added_input_channels) if i == 2 else (0, 0) for i in range(len(x.shape))))
        C = C._replace(cin = C.cin + added_input_channels)
        x = x.movement_op(MovementOps.RESHAPE, (C.bs, C.groups*C.cin, C.iy, C.ix))

      # hack for non multiples of 4 on C.rcout
      added_output_channels = 0
      if C.rcout % 4 != 0 and not (C.rcout == 1 and C.groups%4 == 0):
        added_output_channels = 4 - (C.rcout % 4)
        w = w.movement_op(MovementOps.PAD, tuple((0, added_output_channels) if i == 1 else (0, 0) for i in range(len(w.shape))))
        C = C._replace(rcout = C.rcout + added_output_channels, cout = C.groups * (C.rcout + added_output_channels))

      # packed
      x = x.movement_op(MovementOps.PERMUTE, (0,2,3,1))
      x = x.movement_op(MovementOps.RESHAPE, (C.bs*C.iy, C.ix*C.groups*C.cin//4, 4))

      if C.cin == 1:  # depthwise
        w = w.movement_op(MovementOps.RESHAPE, (C.cout//4,4,C.H*C.W))
        w = w.movement_op(MovementOps.PERMUTE, (0,2,1))
      else:
        w = w.movement_op(MovementOps.RESHAPE, (C.cout//4,4,C.cin//4,4,C.H,C.W))
        w = w.movement_op(MovementOps.PERMUTE, (0,4,2,5,1,3))
        w = w.movement_op(MovementOps.RESHAPE, (C.cout//4, C.H * C.cin//4 * C.W * 4, 4))

      # contiguous creates the image, and early realize static weights
      x, w = x.contiguous(), w.contiguous()
      if get_single_root(w).realized: w.realize()

      # set up the conv from (C.bs*C.iy, C.ix*C.groups*C.cin//4, 4), pad, stride, and expand
      x = x.movement_op(MovementOps.RESHAPE, (C.bs, C.iy, C.ix, C.groups, C.cin))
      x = x.slice(((0, x.shape[0]), (-C.py, x.shape[1]+C.py_), (-C.px, x.shape[2]+C.px_), (0, x.shape[3]), (0, x.shape[4])))
      x = x.movement_op(MovementOps.STRIDED, (
        (C.bs, x.shape[1]*x.shape[2]*C.groups*C.cin),
        (C.oy, C.sy*x.shape[2]*C.groups*C.cin), (C.ox, C.sx*C.groups*C.cin),
        (C.groups, C.cin), (1, 1), (1, 1),
        (C.H, C.dy*x.shape[2]*C.groups*C.cin), (C.W, C.dx*C.groups*C.cin), (C.cin//4 if C.cin >= 4 else 1, 4), (4 if C.cin >= 4 else 1, 1)
      ))
      x = x.movement_op(MovementOps.EXPAND, (C.bs, C.oy, C.ox, C.groups, C.rcout//4 if C.rcout >= 4 else 1, 4 if C.rcout >= 4 else 1, C.H, C.W, x.shape[-2], x.shape[-1]))
      x = x.movement_op(MovementOps.RESHAPE, (C.bs, C.oy, C.ox, C.cout//4, 4, C.H, C.W, x.shape[-2], x.shape[-1]))

      # set up the weights from (C.cout//4, C.H * C.cin//4 * C.W * 4, 4)
      w = w.movement_op(MovementOps.RESHAPE, (C.cout//4, C.H, C.cin//4 if C.cin >= 4 else 1, C.W, 4, 4 if C.cin >= 4 else 1))
      w = w.movement_op(MovementOps.PERMUTE, (0,4,1,3,2,5))
      w = w.movement_op(MovementOps.RESHAPE, (1, 1, 1, C.cout//4, 4, C.H, C.W, w.shape[-2], w.shape[-1]))
      w = w.movement_op(MovementOps.EXPAND, (C.bs, C.oy, C.ox, C.cout//4, 4, C.H, C.W, w.shape[-2], w.shape[-1]))

      # now do the conv in this space and force it to be an image
      ret = x.binary_op(BinaryOps.MUL, w).reduce_op(ReduceOps.SUM, (C.bs, C.oy, C.ox, C.cout//4, 4, 1, 1, 1, 1))
      ret = ret.movement_op(MovementOps.RESHAPE, (C.bs*C.oy, C.ox*C.cout//4, 4)).contiguous()

      # undo hack for non multiples of 4 on C.rcout
      if added_output_channels != 0:
        ret = ret.movement_op(MovementOps.RESHAPE, (C.bs, C.oy, C.ox, C.groups, C.rcout))
        ret = ret.movement_op(MovementOps.SHRINK, tuple((0, s-added_output_channels) if i == 4 else (0, s) for i,s in enumerate(ret.shape)))
        C = C._replace(rcout = C.rcout - added_output_channels, cout = C.groups * (C.rcout - added_output_channels))

      ret = ret.movement_op(MovementOps.RESHAPE, (C.bs, C.oy, C.ox, C.cout))
      ret = ret.movement_op(MovementOps.PERMUTE, (0,3,1,2))
      return ret

    # add padding if the backend can't handle it
    if NOCONV or (not getattr(x.dbuffer, "SUPPORTS_PADDING", False) and not (getattr(x.dbuffer, "SUPPORTS_SIMPLE_PADDING", False) and C.px == C.px_ and C.py == C.py_ and C.px >= 0 and C.py >= 0)):
      x = x.slice(((0, x.shape[0]), (0, x.shape[1]), (-C.py, x.shape[2]+C.py_), (-C.px, x.shape[3]+C.px_)))
      C = C._replace(px=0, px_=0, py=0, py_=0)

    if NOCONV or not issubclass(x.dbuffer, GenericExecAST):
      # universal conv, just mul and reduce
      x = x.movement_op(MovementOps.STRIDED, (
        (C.bs, C.groups*C.cin*x.shape[2]*x.shape[3]), (C.groups, C.cin*x.shape[2]*x.shape[3]),
        (1, 1), (C.oy, C.sy*x.shape[3]), (C.ox, C.sx),
        (C.cin, x.shape[2]*x.shape[3]), (C.H, C.dy*x.shape[3]), (C.W, C.dx)))
      #if C.H <= 3 and C.W <= 3:  # max 9x the RAM overhead, this is im2col
      #  x = x.contiguous()
      x = x.movement_op(MovementOps.EXPAND, (C.bs, C.groups, C.rcout, C.oy, C.ox, C.cin, C.H, C.W))
      w = w.movement_op(MovementOps.RESHAPE, (1, C.groups, C.rcout, 1, 1, C.cin, C.H, C.W)) \
           .movement_op(MovementOps.EXPAND, (C.bs, C.groups, C.rcout, C.oy, C.ox, C.cin, C.H, C.W))
      return x.binary_op(BinaryOps.MUL, w).reduce_op(ReduceOps.SUM, (C.bs, C.groups, C.rcout, C.oy, C.ox, 1, 1, 1)) \
                                          .movement_op(MovementOps.RESHAPE, (C.bs, C.cout, C.oy, C.ox))
    else:
      return LazyBuffer(x.device, C.out_shape, ProcessingOps, LazyOp(op, (x, w), C))

def elementwise_op(op:Union[UnaryOps, BinaryOps], *srcs:LazyBuffer) -> LazyBuffer:
  out_device, out_shape = srcs[0].device, srcs[0].shape

  if MERGE_ELEMENTWISE_OPS or (MERGE_UNARY_OPS and len(set(srcs)) == 1):
    # remove the buffers from any (childless) BinaryOps that feed into this
    srcs = tuple(x.op if x.optype == BinaryOps and len(x.children) == 0 and x.realized is None else x for x in srcs)  # type: ignore

  return LazyBuffer(out_device, out_shape, BinaryOps, LazyOp(op, srcs))
