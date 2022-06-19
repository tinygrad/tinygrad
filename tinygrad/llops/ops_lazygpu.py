import os, time
from tinygrad.llops.ops_lazy import LazyBuffer, LazyOp, find_conv_buf, get_lazybuffers_for_buffer
import functools
#import tinygrad.llops.ops_gpu as gops
import tinygrad.llops.ops_opencl as gops
from tinygrad.ops import ProcessingOps, ReduceOps, BinaryOps, MovementOps, LoadOps, log_op
from tinygrad.shapetracker import ShapeTracker
from tinygrad.helpers import prod
from typing import Tuple, List, Union

from tinygrad.llops.ops_lazy import processing_op, binary_op, unary_op, reduce_op, movement_op

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
  if x.op == ProcessingOps.CONV: code = 'acc'
  else: code = gops.code_for_op[x.op]
  if "A" in code: code = code.replace("A", "("+ast(x.src[0], lazy_srcs)+")")
  if "B" in code: code = code.replace("B", "("+ast(x.src[1], lazy_srcs)+")")
  return code


compile_cache = {}
def compile_binary_op(ret: LazyBuffer, lazy_srcs: List[LazyBuffer]) -> Tuple[str, List[str], List[int]]:
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
      if b.optype == LoadOps and b.shape == (1,) and not st.needs_valid():
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
    __kernel void binop(__global float* restrict res_g"""+(',' if len(opencl_type) >= 2 else '')+', '.join(opencl_type[1:])+""") {
      int gid = get_global_id(0);
      res_g[gid] = _binop("""+', '.join([x.split(" ")[-1].replace("*", "") for x in opencl_type])+""");
    }"""
  lazy_srcs_ret = [lazy_srcs[i] for i in idxs]
  for buf in lazy_srcs_ret:
    inp = buf_st(buf)
    if inp != buf:
      log_op(buf.optype, buf.op.op, buf, [inp], dashed=True)

  real_bufs = [buf_st(x).realize() for x in lazy_srcs_ret]
  gret = gops.GPUBuffer(ret.shape)
  binop = gops.clbuild("binop", prg_src)
  binop([prod(ret.shape)], None, gret.cl, *[x.cl for x in real_bufs])
  return gret, lazy_srcs_ret

@functools.lru_cache(maxsize=None)
def processing_op_compile_hot(prefix_code, middle_code, C, opencl_type):
  cnums = [x for x in list(C[0:12])+[C.dx, C.dy, C.px, C.py]]
  cnames = ["H","W","groups","rcout","cin","oy","ox","iy","ix","ys","xs","bs","dx","dy","px","py"]
  ints = ''.join(f"int {x} = {y};" for x,y in zip(cnames, cnums))
  #print(ints)

  conv_prg = gops.clbuild("conv", prefix_code+"""
  __kernel void conv(__global const float* restrict input, __global const float* restrict weight, __global float* restrict output
    """+(', ' if len(opencl_type) > 0 else '') + ', '.join(opencl_type)+"""
    ) {
    """+ints+"""
    int B = get_global_id(0)/(groups*rcout);  // range 0-bs
    int g = (get_global_id(0)/rcout)%groups;
    int c = get_global_id(0) % rcout;

    int Y = get_global_id(1);  // range 0-oy
    int X = get_global_id(2);  // range 0-ox
    int IY = Y*ys;
    int IX = X*xs;

    int gid = get_global_id(0)*oy*ox + Y*ox + X;

    float acc = 0.0;
    for (int ci = 0; ci < cin; ci++) {
      for (int y = 0; y < H; y++) { for (int x = 0; x < W; x++) {
        int idx_y = y*dy + IY - py;
        int idx_x = x*dx + IX - px;

#ifdef ALLVALID
        acc += input[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + idx_y*ix + idx_x] * \
          weight[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + y*W + x];
#else
        int valid = (idx_y >= 0 && idx_y < iy && idx_x >= 0 && idx_x < ix);
        acc += valid ? input[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + idx_y*ix + idx_x] * \
          weight[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + y*W + x] : 0.0;
#endif
      } }
    }

    // insert binary and unary ops here
    """+middle_code+"""

    output[gid] = acc;
  }""",
  options=tuple(["-DALLVALID"]) if C.px == 0 and C.py == 0 else tuple())
  return conv_prg

def realize_processing_op(ret: LazyBuffer) -> Tuple[gops.GPUBuffer, List[LazyBuffer]]:
  conv = find_conv_buf(ret)
  conv_x, conv_w = conv.src[0], conv.src[1]
  lazy_srcs = list(set(get_lazybuffers_for_buffer(ret)))
  lazy_srcs = [x for x in lazy_srcs if x not in [conv_x, conv_w]]
  prg_src, opencl_type, idxs = compile_binary_op(ret, lazy_srcs)
  lazy_srcs_ret = [lazy_srcs[i] for i in idxs]
  for buf in lazy_srcs_ret:
    inp = buf_st(buf)
    if inp != buf:
      log_op(buf.optype, buf.op.op, buf, [inp], dashed=True)
  real_bufs = [buf_st(x).realize() for x in lazy_srcs_ret]

  #middle_code = "acc = _binop("+', '.join([x.split(" ")[-1].replace("*", "") for x in opencl_type])+");"
  middle_code = "int gid = (outputRow * get_image_width(output) + mad24(startOutputColumn, totalNumPackedOutputChannels, packedOutputChannel))*4;\n"
  vv = "xyzw"
  for i in range(16):
    acc = f"outputValues[{i//4}].{vv[i%4]}"
    args = [x.split(" ")[-1].replace("*", "") for x in opencl_type[2:]]
    args = [acc, f"gid+{i}"]+args
    middle_code += f"{acc} = _binop("+', '.join(args)+");\n"

  C = conv.arg
  gret = gops.GPUBuffer(C.out_shape)

  #conv_prg = processing_op_compile_hot(prg_src, middle_code, C, tuple(opencl_type)[2:])
  #conv_prg([C.bs*C.cout, C.oy, C.ox], None, conv_x.realize().cl, conv_w.realize().cl, gret.cl, *[x.cl for x in real_bufs])
  replacements = {}
  if len(real_bufs) != 0:
    print(prg_src)
    print(real_bufs)
    print(middle_code)
    print(tuple(opencl_type)[2:])
    replacements["//PREFIX"] = prg_src
    replacements["//ARGS"] = ","+','.join(tuple(opencl_type)[2:])
    replacements["//BINOP"] = middle_code
    #assert False

  gops.conv(conv_x.realize(), conv_w.realize(), gret, C, replacements, real_bufs)

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
    elif self.optype == ReduceOps:
      lazy_srcs = [self.op.src[0]]
      ret = gops.reduce_op(self.op.op, self.op.src[0].realize(), self.op.arg)
    elif self.optype == MovementOps:
      root, st = movementop_buf(self)
      lazy_srcs += [root]
      # NOTE: contiguous can have the wrong shape
      #if st.contiguous: ret = gops.GPUBuffer(st.shape, root.realize())
      #else: ret = gops.contiguous(root.realize(), st)
      ret = gops.contiguous(root.realize(), st)
    self.realized = ret

    if self.SHOULD_LOG:
      log_op(self.optype, self.op.op, self, lazy_srcs)
    return self.realized

  @staticmethod
  def fromCPU(x):
    return LazyGPUBuffer(x.shape, LoadOps, LazyOp(LoadOps.FROMCPU, [], x))

  def toCPU(self):
    #return self.realize().toCPU()

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
LazyOpenCLBuffer = LazyGPUBuffer
