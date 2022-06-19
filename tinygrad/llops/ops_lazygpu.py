import os, time
from tinygrad.llops.ops_lazy import LazyBuffer, LazyOp, find_conv_buf, get_lazybuffers_for_buffer
import functools
import tinygrad.llops.ops_gpu as gops
from tinygrad.ops import ProcessingOps, UnaryOps, BinaryOps, MovementOps, LoadOps, log_op
from tinygrad.shapetracker import ShapeTracker
from tinygrad.helpers import prod
from typing import Tuple, List, Union

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

@functools.lru_cache(maxsize=None)
def movementop_buf(x: LazyBuffer):
  return movementop_st(x.op)

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
    x, xst = movementop_buf(x)
  else:
    xst = ShapeTracker(*x.shape)
  return x, xst

# TODO: refactor with the above two
@functools.lru_cache(maxsize=None)
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
  elif op == ProcessingOps.CONV: code = 'acc'
  if "A" in code: code = code.replace("A", "("+ast(x.src[0], lazy_srcs)+")")
  if "B" in code: code = code.replace("B", "("+ast(x.src[1], lazy_srcs)+")")
  return code


compile_cache = {}
def compile_binary_op(ret: LazyBuffer, lazy_srcs: List[LazyBuffer]) -> Tuple[str, list[str], list[int]]:
  if ret not in compile_cache:
    lazy_srcs_st : List[Tuple[LazyBuffer, ShapeTracker]] = [to_st(x) for x in lazy_srcs]
    opencl_type = []
    if ret.optype == ProcessingOps:
      opencl_type.append("float acc")
    opencl_type.append('int gid')
    opencl_src = []
    opencl_interior_src = []
    idxs = []
    for argn,(b,st) in enumerate(lazy_srcs_st):
      if b.optype == None and b.shape == (1,) and not st.needs_valid():
        opencl_interior_src.append(f"float arg_{argn} = {b.op.arg[0]};")
      else:
        opencl_src.append("""inline float get_"""+str(argn)+"""(__global const float* restrict x, int idx) {
          """+("int valid = 1;" if st.needs_valid() else "")+"""
          """+st.expr().replace('//', '/')+""";
          """+("return valid ? x[idx] : 0.0;" if st.needs_valid() else "return x[idx];")+"""
        }""")
        opencl_interior_src.append(f"float arg_{argn} = get_{argn}(buf_{argn}, gid);")
        opencl_type.append(f"__global const float* restrict buf_{argn}")
        idxs.append(argn)

    prg_src = '\n'.join(opencl_src)+"""
    inline float _binop("""+', '.join(opencl_type)+""") {
      """+'\n'.join(opencl_interior_src)+"""
      return """+ast(ret.op, lazy_srcs)+""";
    }"""
    compile_cache[ret] = (prg_src, opencl_type, idxs)
  return compile_cache[ret]


def realize_binary_op(ret: LazyBuffer) -> Tuple[gops.GPUBuffer, List[LazyBuffer]]:
  lazy_srcs = list(set(get_lazybuffers_for_buffer(ret)))
  prg_src, opencl_type, idxs = compile_binary_op(ret, lazy_srcs)
  prg_src += """
    __kernel void binop(__global float* restrict res_g, """+', '.join(opencl_type[1:])+""") {
      int gid = get_global_id(0);
      res_g[gid] = _binop("""+', '.join([x.split(" ")[-1].replace("*", "") for x in opencl_type])+""");
    }"""
  lazy_srcs_ret = [buf_st(lazy_srcs[i]) for i in idxs]
  real_bufs = [x.realize() for x in lazy_srcs_ret]

  gret = gops.GPUBuffer(ret.shape)
  binop = gops.clbuild("binop", prg_src)
  binop([prod(ret.shape)], None, gret.cl, *[x.cl for x in real_bufs])
  return gret, lazy_srcs_ret

def realize_processing_op(ret: LazyBuffer) -> Tuple[gops.GPUBuffer, List[LazyBuffer]]:
  conv = find_conv_buf(ret)
  conv_x, conv_w = conv.src[0], conv.src[1]
  lazy_srcs = list(set(get_lazybuffers_for_buffer(ret)))
  lazy_srcs = [x for x in lazy_srcs if x not in [conv_x, conv_w]]
  prg_src, opencl_type, idxs = compile_binary_op(ret, lazy_srcs)
  lazy_srcs_ret = [buf_st(lazy_srcs[i]) for i in idxs]
  real_bufs = [x.realize() for x in lazy_srcs_ret]

  middle_code = "acc = _binop("+', '.join([x.split(" ")[-1].replace("*", "") for x in opencl_type])+");"

  gret = gops.processing_op(conv.op, conv_x.realize(), conv_w.realize(), conv.arg,
    prg_src, middle_code, real_bufs, opencl_type[2:])
  return gret, lazy_srcs_ret+[conv_x, conv_w]

realized_buffers = []
class LazyGPUBuffer(LazyBuffer):
  SHOULD_LOG = True
  def realize(self:LazyBuffer) -> gops.GPUBuffer:
    if self.realized is not None: return self.realized
    realized_buffers.append(self)

    lazy_srcs = []
    ret = None
    if self.optype == LoadOps:
      ret = gops.GPUBuffer(self.shape, self.op.arg)
    elif self.optype == ProcessingOps:
      ret, lazy_srcs = realize_processing_op(self)
    elif self.optype == BinaryOps:
      ret, lazy_srcs = realize_binary_op(self)
    elif self.optype == MovementOps:
      root, st = movementop_buf(self)
      lazy_srcs += [root]
      if st.contiguous: ret = root.realize()
      else: ret = gops.contiguous(root.realize(), st)
    self.realized = ret

    if self.op.op is not None and self.SHOULD_LOG:
      opname = self.optype.__name__ if self.optype is not None else 'load'
      log_op(opname, get_lazyops(self.op), self, lazy_srcs)
    return self.realized

  @staticmethod
  def fromCPU(x):
    return LazyGPUBuffer(x.shape, LoadOps, LazyOp(LoadOps.FROMCPU, [], x))

  def toCPU(self):
    global realized_buffers
    # for the kernel builds to not count in timing
    junk = self.realize().toCPU()
    print("derealizing %d" % len(realized_buffers))
    for b in realized_buffers:
      if b.optype != LoadOps:
        b.realized = None
    realized_buffers = []

    LazyGPUBuffer.SHOULD_LOG = False
    if int(os.getenv("PROFILE", 0)) == 1:
      import cProfile
      import pstats, io
      from pstats import SortKey
      import io
      pr = cProfile.Profile(timer=time.perf_counter_ns, timeunit=1e-6)
      pr.enable()
      self.realize()
      pr.disable()
      ps = pstats.Stats(pr).sort_stats(SortKey.TIME)
      ps.print_stats()

    st = time.monotonic()
    ret = self.realize()
    mt = time.monotonic()
    ret = ret.toCPU()
    et = time.monotonic()

    print(f"realized in {(et-st)*1000:.2f} ms, waited {(et-mt)*1000:.2f} ms for kernels ({(mt-st)*1000:.2f} ms in python)")
    return ret