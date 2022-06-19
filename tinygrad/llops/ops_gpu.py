import functools
import numpy as np
import pyopencl as cl
from tinygrad.helpers import prod
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps
from tinygrad.shapetracker import ShapeTracker, View, strides_for_shape

cl_ctx, cl_queue = None, None
def get_cl_ctx(): return cl_ctx
def get_cl_queue(): return cl_queue
def require_init_gpu():
  global cl_ctx, cl_queue
  if cl_ctx is None:
    devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
    if len(devices) == 0:  # settle for CPU
      devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.CPU)
    cl_ctx = cl.Context(devices=devices)
    cl_queue = cl.CommandQueue(cl_ctx)  # this is an in-order command queue

def roundup(x, n=4): return (x+(n-1))//n * n

class GPUBuffer:
  def __init__(self, shape, hostbuf=None):
    require_init_gpu()
    self.shape = tuple(shape)
    self.cl = hostbuf.cl if isinstance(hostbuf, GPUBuffer) else cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, 4*roundup(prod(self.shape)))  # padding
    if hostbuf is not None and not isinstance(hostbuf, GPUBuffer):
      cl.enqueue_copy(cl_queue, self.cl, hostbuf.astype(np.float32).ravel(), is_blocking=False)

  @property
  def dtype(self): return np.float32

  def __repr__(self):
    return f"<GPUBuffer with shape {self.shape!r}>"

  @staticmethod
  def fromCPU(x):
    return GPUBuffer(x.shape, x.view(np.ndarray))

  def toCPU(self):
    data = np.empty(self.shape, dtype=np.float32)
    cl.enqueue_copy(cl_queue, data, self.cl, is_blocking=True)
    return data

class CLProgram:
  def __init__(self, name, prg, options, argdtypes):
    self.built = cl.Program(cl_ctx, prg).build(options=options)
    self.clprg = self.built.__getattr__(name)
    if argdtypes is not None: self.clprg.set_scalar_arg_dtypes(argdtypes)
  def __call__(self, *args): self.clprg(cl_queue, *args)

@functools.lru_cache(maxsize=None)
def clbuild(name, prg, options=tuple(), argdtypes=None):
  #print("cache miss")
  #print(prg)
  return CLProgram(name, prg, options, argdtypes)

code_for_op = {
  UnaryOps.RELU: 'max(A, (float)0.)', UnaryOps.EXP: 'exp(A)', UnaryOps.LOG: 'log(A)', UnaryOps.NEG: '-A', UnaryOps.SIGN: 'sign(A)',
  BinaryOps.ADD: "A+B", BinaryOps.SUB: "A-B", BinaryOps.MUL: "A*B", BinaryOps.DIV: "B/A", BinaryOps.POW: "pow(A,B)", BinaryOps.CMPEQ: "(A==B)"
}

def unary_op(op, x):
  Buffer = x.__class__
  ret = Buffer(x.shape)
  unop = clbuild("unop", """
  __kernel void unop(__global const float4 *a_g, __global float4 *res_g) {
    int gid = get_global_id(0);
    float4 A = a_g[gid];
    res_g[gid] = convert_float4("""+code_for_op[op]+""");
  }""")
  unop([roundup(prod(ret.shape))//4], None, x.cl, ret.cl)
  return ret

def binary_op(op, x, y):
  Buffer = x.__class__
  ret = Buffer(x.shape)
  assert x.shape == ret.shape and y.shape == ret.shape
  binop = clbuild("binop", """
  __kernel void binop(__global const float4 *a_g, __global const float4 *b_g, __global float4 *res_g) {
    int gid = get_global_id(0);
    float4 A = a_g[gid];
    float4 B = b_g[gid];
    res_g[gid] = convert_float4("""+code_for_op[op]+""");
  }""")
  binop([roundup(prod(ret.shape))//4], None, x.cl, y.cl, ret.cl)
  return ret

def reduce_op(op, inp, new_shape):
  Buffer = x.__class__
  ret = Buffer(new_shape)
  if op == ReduceOps.SUM: code, start = "out += a", "0.0"
  elif op == ReduceOps.MAX: code, start = "out = max(a,out)", "-INFINITY"
  else: raise Exception(f"{op} isn't supported")

  # reverse operation of expand, this validates inputs
  st = ShapeTracker(*ret.shape).movement_op(MovementOps.EXPAND, inp.shape)
  # this takes a ret index to an inp index, indexing 0 on the reduced strides
  view = View(ret.shape, strides_for_shape(inp.shape))

  # generate loops with combined adjacent reduce axis
  acc = 1
  loop_start, loop_end = [], []
  for shp,stride in st.views[-1].shape_strides[::-1]:
    if stride == 0:
      loop_start.append(f"for (int axis_{len(loop_start)} = 0; axis_{len(loop_start)} < {shp}; axis_{len(loop_start)}++) {{")
      loop_end.append(f"idx += {acc}; }} idx -= {shp*acc};")
    acc *= shp

  # TODO: support multistage reduces
  prg = """
  __kernel void reduce(__global const float *a_g, __global float *res_g) {
    int gid = get_global_id(0); int idx = gid;"""+view.expr.replace('//', '/')+""";
    float out = """+start+""";\n"""+ \
      '\n'.join(loop_start[::-1])+"""
        float a = a_g[idx];
        """+code+""";\n"""+ \
      '\n'.join(loop_end)+"""
    res_g[gid] = out;
  }"""
  clbuild("reduce", prg)([prod(ret.shape)], None, inp.cl, ret.cl)
  return ret

def contiguous(x, st, ret=None):
  Buffer = x.__class__
  if ret is None: ret = Buffer(st.shape)
  clbuild("contiguous", """__kernel void contiguous(__global const float *x, __global float *ret) {
    int gid = get_global_id(0); int valid = 1; int idx = gid; """+st.expr().replace('//', '/')+""";
    ret[gid] = valid ? x[idx] : 0.0;  // should never be out-of-bounds accesses
  }""")([prod(ret.shape)], None, x.cl, ret.cl)
  return ret

def movement_op(op, x, arg=None):
  Buffer = x.__class__
  st = ShapeTracker(*x.shape).movement_op(op, arg)
  if st.contiguous: return Buffer(st.shape, x)
  else: return contiguous(x, st)

def processing_op(op,x,w,C):
  Buffer = x.__class__
  ret = Buffer((C.bs, C.cout, C.oy, C.ox))
  assert op == ProcessingOps.CONV, f"{op} isn't supported"
  conv_prg = clbuild("conv", """
  __kernel void conv(__global const float* restrict input, __global const float* restrict weight, __global float* restrict output,
    int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs, int bs, int dx, int dy, int px, int py) {
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

#ifdef ONEBYONE
      acc += input[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + IY*ix + IX] * \
        weight[g*rcout*cin + c*cin + ci];
#else
      for (int y = 0; y < H; y++) { for (int x = 0; x < W; x++) {
        int idx_y = y*dy + IY - py;
        int idx_x = x*dx + IX - px;
        int valid = (idx_y >= 0 && idx_y < iy && idx_x >= 0 && idx_x < ix);
        acc += valid ? input[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + idx_y*ix + idx_x] * \
          weight[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + y*W + x] : 0.0;
      } }
#endif
    }
    output[gid] = acc;
  }""",
  options=tuple(["-DONEBYONE"]) if C.H == 1 and C.W == 1 and C.px == 0 and C.py == 0 else tuple(),
  argdtypes=tuple([None, None, None] + [np.int32]*16))
  conv_prg([C.bs*C.cout, C.oy, C.ox], None, x.cl, w.cl, ret.cl,
    *[x for x in list(C[0:12])+[C.dx, C.dy, C.px, C.py]])
  return ret
