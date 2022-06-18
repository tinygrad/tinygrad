from __future__ import annotations
from tinygrad.shapetracker import ShapeTracker
from collections import namedtuple
import functools
import numpy as np
from tinygrad.helpers import prod
import sys
import time
sys.setrecursionlimit(10000)

# TODO: these aren't really ops
from typing import Union, NamedTuple, List, Any, Tuple
from tinygrad.ops import ReduceOps, BinaryOps, MovementOps, ProcessingOps, UnaryOps, log_op
Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps]
ElementWiseOps = Union[UnaryOps, BinaryOps]
OpTypes = Union[ElementWiseOps, ReduceOps, MovementOps, ProcessingOps]

# sequential movement ops can be flattened into 1 movement op
MERGE_MOVEMENT_OPS = True

# movement ops can be moved above elementwise ops 
SHUFFLE_MOVEMENT_OPS = True

# if you stick the right movement ops, they might disappear!
# TODO: this is wrong, improve ShapeTracker
REMOVE_MOVEMENT_NOPS = False

# "sequential" elementwise ops can be merged into 1 big elementwise op
MERGE_ELEMENTWISE_OPS = True

# after the conv is done, it can run elementwise ops on its output
MERGE_ELEMENTWISE_INTO_CONV_OUTPUT = False

class LazyOp(NamedTuple):
  op: Op
  src: List[Union[LazyOp, LazyBuffer]]
  arg: Any = None

def get_lazybuffers(op:LazyOp):
  #print(op)
  ret = []
  for x in op.src:
    if isinstance(x, LazyOp):
      ret += get_lazybuffers(x)
    elif isinstance(x, LazyBuffer):
      ret.append(x)
    else: raise Exception("wtf")
  return ret

def get_lazyops(op:LazyOp):
  ret = [op.op]
  for x in op.src:
    if isinstance(x, LazyOp):
      ret += get_lazyops(x)
    elif isinstance(x, LazyBuffer):
      pass
    else:
      raise Exception("wtf")
  return ret

import tinygrad.llops.ops_gpu as gops

def movementop_st(root: LazyOp) -> Tuple[LazyBuffer, ShapeTracker]:
  op_arg = []
  while isinstance(root, LazyOp):
    op_arg.append((root.op, root.arg))
    root = root.src[0]
  assert isinstance(root, LazyBuffer)
  st = ShapeTracker(*root.shape)
  for o,a in op_arg[::-1]:
    st = st.movement_op(o, a)
  return root, st

def to_st(x: LazyBuffer) -> Tuple[LazyBuffer, ShapeTracker]:
  if x.optype == MovementOps:
    x, xst = movementop_st(x.op)
  else:
    xst = ShapeTracker(*x.shape)
  return x, xst

# TODO: refactor with the above two
def buf_st(x: LazyBuffer) -> LazyBuffer:
  if x.optype == MovementOps:
    x = x.op
    while isinstance(x, LazyOp):
      x = x.src[0]
  return x

def ast(x: Union[LazyBuffer, LazyOp], lazy_srcs: List[LazyBuffer]) -> str:
  if isinstance(x, LazyBuffer):
    return f"arg_{lazy_srcs.index(x)}"
  # it's an op
  op = x.op
  if op == BinaryOps.ADD: code = "A+B"
  elif op == BinaryOps.SUB: code = "A-B"
  elif op == BinaryOps.MUL: code = "A*B"
  elif op == BinaryOps.DIV: code = "B/A"
  elif op == BinaryOps.POW: code = "pow(A,B)"
  elif op == BinaryOps.CMPEQ: code = "1.0f*(A==B)"
  elif op == UnaryOps.RELU: code = 'max(A, (float)0.)'
  elif op == UnaryOps.EXP: code = 'exp(A)'
  elif op == UnaryOps.LOG: code = 'log(A)'
  elif op == UnaryOps.NEG: code = '-A'
  elif op == UnaryOps.SIGN: code = 'sign(A)'
  code = code.replace("A", "("+ast(x.src[0], lazy_srcs)+")")
  if len(x.src) > 1:
    code = code.replace("B", "("+ast(x.src[1], lazy_srcs)+")")
  return code

compile_cache = {}
def compile_binary_op(ret: LazyBuffer, lazy_srcs: List[LazyBuffer]) -> Tuple[str, list[str], list[int]]:
  if ret not in compile_cache:
    lazy_srcs_st : List[Tuple[LazyBuffer, ShapeTracker]] = [to_st(x) for x in lazy_srcs]
    opencl_type = ["__global float *res_g"]
    opencl_src = []
    opencl_interior_src = []
    idxs = []
    for argn,(b,st) in enumerate(lazy_srcs_st):
      if b.optype == None and b.shape == (1,) and not st.needs_valid():
        opencl_interior_src.append(f"float arg_{argn} = {b.op.arg[0]};")
      else:
        opencl_src.append("""inline float get_"""+str(argn)+"""(__global const float *x, int idx) {
          int valid = 1;
          """+st.expr().replace('//', '/')+""";
          """+("return valid ? x[idx] : 0.0;" if st.needs_valid() else "return x[idx];")+"""
        }""")
        opencl_interior_src.append(f"float arg_{argn} = get_{argn}(buf_{argn}, gid);")
        opencl_type.append(f"__global const float *buf_{argn}")
        idxs.append(argn)

    prg_src = '\n'.join(opencl_src)+"""
    inline void _binop("""+', '.join(opencl_type)+""") {
      int gid = get_global_id(0);
      """+'\n'.join(opencl_interior_src)+"""
      res_g[gid] = """+ast(ret.op, lazy_srcs)+""";
    }"""
    compile_cache[ret] = (prg_src, opencl_type, idxs)
  return compile_cache[ret]

def realize_binary_op(ret: LazyBuffer) -> Tuple[gops.GPUBuffer, List[LazyBuffer]]:
  lazy_srcs = list(set(get_lazybuffers(ret.op)))
  prg_src, opencl_type, idxs = compile_binary_op(ret, lazy_srcs)
  prg_src += """
    __kernel void binop("""+', '.join(opencl_type)+""") {
      _binop("""+', '.join([x.split("*")[1] for x in opencl_type])+""");
    }"""
  lazy_srcs_ret = [buf_st(lazy_srcs[i]) for i in idxs]
  real_bufs = [x.realize() for x in lazy_srcs_ret]

  #print(prg_src)
  gret = gops.GPUBuffer(ret.shape)
  binop = gops.clbuild("binop", prg_src)
  binop([prod(ret.shape)], None, gret.cl, *[x.cl for x in real_bufs])
  return gret, lazy_srcs_ret

realized_buffers = []
class LazyBuffer:
  def __init__(self, shape:tuple, optype:OpTypes, op:LazyOp):
    assert isinstance(op, LazyOp)
    assert isinstance(op.src, list)
    assert isinstance(shape, tuple)

    #print(shape, optype, op)
    self.shape = shape
    self.optype = optype
    self.dtype = np.float32
    self.op = op
    self.realized = None

    if self.optype == None:
      # nonlazy create from hostbuf
      self.realized = gops.GPUBuffer(self.shape, self.op.arg)

  @property
  def opname(self):
    return self.optype.__name__ if self.optype is not None else 'load'

  def __repr__(self):
    return f"<LB {self.opname}: {self.op.op}>"

  def realize(self:LazyBuffer) -> gops.GPUBuffer:
    if self.optype is not None: realized_buffers.append(self)
    if self.realized is not None:
      return self.realized
    lazy_srcs = []
    ret = None
    if self.optype == None:
      # created from hostbuf
      assert False
      ret = gops.GPUBuffer(self.shape, self.op.arg)
    elif self.optype == ProcessingOps:
      x,w = self.op.src
      lazy_srcs = [x,w]
      ret = gops.processing_op(self.op.op, x.realize(), w.realize(), self.op.arg)
    elif self.optype == BinaryOps:
      ret, lazy_srcs = realize_binary_op(self)
    elif self.optype == MovementOps:
      root, st = movementop_st(self.op)
      lazy_srcs += [root]
      ret = gops.contiguous(root.realize(), st)
    self.realized = ret

    if self.op.op is not None: log_op(self.opname, get_lazyops(self.op), self, lazy_srcs)
    return self.realized

  @staticmethod
  def fromCPU(x):
    return LazyBuffer(x.shape, None, LazyOp(None, [], x))

  def toCPU(self):
    global realized_buffers
    # for the kernel builds to not count in timing
    junk = self.realize().toCPU()
    print("derealizing %d" % len(realized_buffers))
    for b in realized_buffers:
      b.realized = None
    realized_buffers = []

    """
    import cProfile
    from pstats import SortKey
    cProfile.runctx("self.realize()", globals(), locals(), sort=SortKey.TIME)
    """

    st = time.monotonic()
    ret = self.realize()
    mt = time.monotonic()
    ret = ret.toCPU()
    et = time.monotonic()

    print(f"realized in {(et-st)*1000:.2f} ms, waited {(et-mt)*1000:.2f} ms for kernels ({(mt-st)*1000:.2f} ms in python)")
    return ret

@functools.lru_cache(maxsize=None)
def elementwise_op(op, srcs:Tuple[LazyBuffer]) -> LazyBuffer:
  out_shape = srcs[0].shape
  if MERGE_ELEMENTWISE_INTO_CONV_OUTPUT:
    # TODO: this is wrong
    cnt = sum([x.optype == ProcessingOps for x in srcs])
    if cnt == 1:
      srcs = [x.op if x.optype == ProcessingOps else x for x in srcs]
      return LazyBuffer(out_shape, ProcessingOps, LazyOp(op, srcs))
    elif cnt == 2:
      # have to confirm they are the same conv
      def find_conv(x:LazyOp):
        if isinstance(x, LazyBuffer):
          return None
        if isinstance(x.op, ProcessingOps):
          return x
        for s in x.src:
          tst = find_conv(s)
          if tst is not None:
            return tst
        return None
      c1 = find_conv(srcs[0].op)
      c2 = find_conv(srcs[1].op)
      #print(c1.op, c2.op)
      if c1.arg == c2.arg and tuple(c1.src) == tuple(c2.src):
        srcs = [x.op if x.optype == ProcessingOps else x for x in srcs]
        return LazyBuffer(out_shape, ProcessingOps, LazyOp(op, srcs))
      else:
        #@functools.lru_cache
        memo = {}
        def depends(op:LazyOp, needle:LazyBuffer) -> bool:
          nonlocal memo
          if id(op) in memo: return memo[id(op)]
          bufs = get_lazybuffers(op)
          if needle in bufs:
            memo[id(op)] = True
            return True
          ret = False
          for b in bufs:
            if depends(b.op, needle):
              ret = True
              break
          memo[id(op)] = ret
          return ret
        if depends(srcs[0].op, srcs[1]):
          srcs = [srcs[0].op, srcs[1]]
        elif depends(srcs[1].op, srcs[0]):
          srcs = [srcs[0], srcs[1].op]
        else:
          # all three are okay
          #return LazyBuffer(out_shape, BinaryOps, LazyOp(op, list(srcs)))
          srcs = [srcs[0].op, srcs[1]]
          #srcs = [srcs[0], srcs[1].op]
        return LazyBuffer(out_shape, ProcessingOps, LazyOp(op, srcs))

  if MERGE_ELEMENTWISE_OPS:
    srcs = [x.op if x.optype == BinaryOps else x for x in srcs]
  return LazyBuffer(out_shape, BinaryOps, LazyOp(op, list(srcs)))

# caching is safe here, the same op and arg applied to the same buffer is the same
@functools.lru_cache(maxsize=None)
def movement_op(op:MovementOps, x:LazyBuffer, arg) -> LazyBuffer:
  st = ShapeTracker(*x.shape)
  st = st.movement_op(op, arg)
  if len(st.views) == 1: return x    # this is a no-op

  if SHUFFLE_MOVEMENT_OPS:
    if x.optype == BinaryOps:
      def replace_w_movement_op(y:Union[LazyOp, LazyBuffer]) -> LazyBuffer:
        if isinstance(y, LazyBuffer):
          return movement_op(op, y, arg)
        elif isinstance(y, LazyOp):
          return elementwise_op(y.op, tuple([replace_w_movement_op(z) for z in y.src]))
      return replace_w_movement_op(x.op)

  if REMOVE_MOVEMENT_NOPS:
    if x.optype == MovementOps:
      root = x.op
      op_arg = [(op, arg)]
      while isinstance(root, LazyOp):
        op_arg.append((root.op, root.arg))
        root = root.src[0]
      assert isinstance(root, LazyBuffer)
      rst = ShapeTracker(*root.shape)
      for o,a in op_arg[::-1]:
        rst = rst.movement_op(o, a)
      # TODO: this check is wrong, we used the shapetracker for a reason
      if rst.shape == root.shape:
        return root

  if MERGE_MOVEMENT_OPS:
    if isinstance(x.op.op, MovementOps):
      x = x.op

  # otherwise just create the movement op
  return LazyBuffer(st.shape, MovementOps, LazyOp(op, [x], arg))

def reduce_op(op, x, new_shape): return LazyBuffer(new_shape, ReduceOps, LazyOp(op, [x], new_shape))
def processing_op(op, x, w, C): return LazyBuffer(C.out_shape, ProcessingOps, LazyOp(op, [x, w], C))

# universal dispatcher?
class Ops:
  def unary_op(ctx, op, x): return elementwise_op(op, (x,))
  def binary_op(ctx, op, x, y): return elementwise_op(op, (x,y))
  def movement_op(ctx, op, x, arg): return movement_op(op, x, tuple(arg))
  # blocker ops
  def reduce_op(ctx, op, x, new_shape): return reduce_op(op, x, new_shape)
  def processing_op(ctx, op, x, w, C): return processing_op(op, x, w, C)
