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

i32 = np.int32
def roundup(x, n=4): return (x+(n-1))//n * n
def sync(): cl_queue.finish()

class GPUBuffer:
  def __init__(self, shape, hostbuf=None):
    require_init_gpu()
    self.shape, self.dtype = tuple(shape), np.float32
    self.cl = hostbuf.cl if isinstance(hostbuf, GPUBuffer) else cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, 4*roundup(prod(shape)))  # padding
    if hostbuf is not None and not isinstance(hostbuf, GPUBuffer):
      cl.enqueue_copy(cl_queue, self.cl, hostbuf.astype(np.float32).ravel())

  def __repr__(self):
    return f"<GPUBuffer with shape {self.shape!r}>"

  @staticmethod
  def fromCPU(x):
    return GPUBuffer(x.shape, x.view(np.ndarray))

  def toCPU(self):
    data = np.empty(self.shape, dtype=np.float32)
    sync()
    cl.enqueue_copy(cl_queue, data, self.cl, is_blocking=True)
    return data

@functools.lru_cache
def clbuild(name, prg, options=tuple()):
  clprg = cl.Program(cl_ctx, prg).build(options=options).__getattr__(name)
  def run(*args): clprg(cl_queue, *args)
  return run

def unary_op(ctx, op, x):
  ret = ctx.buffer(x.shape)
  if op == UnaryOps.RELU: code = 'max(a, (float)0.)'
  elif op == UnaryOps.EXP: code = 'exp(a)'
  elif op == UnaryOps.LOG: code = 'log(a)'
  elif op == UnaryOps.NEG: code = '-a'
  elif op == UnaryOps.SIGN: code = 'sign(a)'
  else: raise Exception(f"{op} isn't supported")
  unop = clbuild("unop", """
  __kernel void unop(__global const float4 *a_g, __global float4 *res_g) {
    int gid = get_global_id(0);
    float4 a = a_g[gid];
    res_g[gid] = """+code+""";
  }""")
  unop([roundup(prod(ret.shape))//4], None, x.cl, ret.cl)
  return ret

def binary_op(ctx, op, x, y):
  ret = ctx.buffer(x.shape)
  if op == BinaryOps.ADD: code = "a+b"
  elif op == BinaryOps.SUB: code = "a-b"
  elif op == BinaryOps.MUL: code = "a*b"
  elif op == BinaryOps.DIV: code = "b/a"
  elif op == BinaryOps.POW: code = "pow(a,b)"
  elif op == BinaryOps.CMPEQ: code = "(float4)(1.0f*(a.x==b.x), 1.0f*(a.y==b.y), 1.0f*(a.z==b.z), 1.0f*(a.w==b.w))"
  else: raise Exception(f"{op} isn't supported")
  assert x.shape == ret.shape and y.shape == ret.shape
  binop = clbuild("binop", """
  __kernel void binop(__global const float4 *a_g, __global const float4 *b_g, __global float4 *res_g) {
    int gid = get_global_id(0);
    float4 a = a_g[gid];
    float4 b = b_g[gid];
    res_g[gid] = """+code+""";
  }""")
  binop([roundup(prod(ret.shape))//4], None, x.cl, y.cl, ret.cl)
  return ret

def reduce_op(ctx, op, inp, new_shape):
  ret = ctx.buffer(new_shape)
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

def contiguous(ctx, x, st):
  ret = ctx.buffer(st.shape)
  clbuild("contiguous", """__kernel void contiguous(__global const float *x, __global float *ret) {
    int gid = get_global_id(0); int valid = 1; int idx = gid; """+st.expr().replace('//', '/')+""";
    ret[gid] = valid ? x[idx] : 0.0;  // should never be out-of-bounds accesses
  }""")([prod(ret.shape)], None, x.cl, ret.cl)
  return ret

def movement_op(ctx, op, x, arg=None):
  return contiguous(ctx, x, ShapeTracker(*x.shape).movement_op(op, arg))

def processing_op(ctx,op,x,w,out_shape,C):
  ret = ctx.buffer(out_shape)
  assert op == ProcessingOps.CONV, f"{op} isn't supported"
  # input  = (bs, groups, cin, iy, ix)
  # weight = (groups, rcout, cin, H, W)
  # output = (bs, groups, rcout, oy, ox)
  conv_prg = clbuild("conv", """
  __kernel void conv(__global const float *input, __global const float *weight, __global float *output,
    int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs, int bs, int dx, int dy, int px, int py) {

    int B = get_global_id(0)/(groups*rcout);  // range 0-bs
    int g = (get_global_id(0)/rcout)%groups;
    int c = get_global_id(0) % rcout;

    int Y = get_global_id(1);  // range 0-oy
    int X = get_global_id(2);  // range 0-ox
    int IY = Y*ys;
    int IX = X*xs;

    float acc = 0.0;
    for (int ci = 0; ci < cin; ci++) {
      for (int y = 0; y < H; y++) { for (int x = 0; x < W; x++) {
        int idx_y = y*dy + IY - py;
        int idx_x = x*dx + IX - px;
        int valid = (idx_y >= 0 && idx_y < iy && idx_x >= 0 && idx_x < ix);
        acc += valid ? input[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + idx_y*ix + idx_x] * \
          weight[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + y*W + x] : 0.0;
      } }
    }
    output[B*groups*rcout*oy*ox + g*rcout*oy*ox + c*oy*ox + Y*ox + X] = acc;
  }""")

  conv_prg([C.bs*C.groups*C.rcout, C.oy, C.ox], None, x.cl, w.cl, ret.cl, *[i32(x) for x in list(C[0:12])+[C.dx, C.dy, C.px, C.py]])
  return ret
