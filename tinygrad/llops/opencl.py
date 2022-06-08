# llops don't know about derivatives or memory allocation
import functools
import numpy as np
import pyopencl as cl
from tinygrad.helpers import binary_broadcast, UnaryOps, BinaryOps, ReduceOps

cl_ctx, cl_queue = None, None
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
    self.cl = hostbuf.cl if isinstance(hostbuf, GPUBuffer) else cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, 4*roundup(np.prod(shape)))  # padding
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

def buffer_np(x):
  return cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)

@functools.lru_cache
def clbuild(name, prg):
  clprg = cl.Program(cl_ctx, prg).build().__getattr__(name)
  def run(*args): clprg(cl_queue, *args)
  return run

def unary_op(op, x, ret):
  if op == UnaryOps.RELU: code = 'max(a, (float)0.)'
  elif op == UnaryOps.EXP: code = 'exp(a)'
  elif op == UnaryOps.LOG: code = 'log(a)'
  elif op == UnaryOps.NEG: code = '-a'
  else: raise Exception(f"{op} isn't supported")
  unop = clbuild("unop", """
  __kernel void unop(__global const float4 *a_g, __global float4 *res_g) {
    int gid = get_global_id(0);
    float4 a = a_g[gid];
    res_g[gid] = """+code+""";
  }""")
  unop([roundup(np.prod(ret.shape))//4], None, x.cl, ret.cl)
  return ret

@functools.lru_cache
def get_binop_prg(code, complist):
  ndims = len(complist)
  args = "".join([f", int d{i}" for i in range(ndims)] + [f", int p{i}" for i in range(ndims-1)])
  compute_idx_rets = "".join([f"\n    int idx_ret{i} = (gid0 / {f'p{i}' if i < ndims-1 else '1'}) % d{i};" for i in range(ndims)])

  idx_exprs = ["0", "0"] # [idx_x, idx_y]
  for i in range(ndims):
    for j in range(2):
      if complist[i][j]:
        idx_exprs[j] = "idx_ret%d + d%d*(%s)" % (i, i, idx_exprs[j])

  dtype = ["float", "float", "float"]
  prg = """__kernel void binop(__global const """+dtype[0]+""" *x_g, __global const """+dtype[1]+""" *y_g, __global """+dtype[2]+""" *res_g"""+args+""") {
    int gid0 = get_global_id(0);"""+compute_idx_rets+"""
    """+dtype[0]+""" a = x_g["""+idx_exprs[0]+"""];
    """+dtype[1]+""" b = y_g["""+idx_exprs[1]+"""];
    res_g[gid0] = """+code+""";\n}"""
  return cl.Program(cl_ctx, prg).build(), dtype[2] == "float4"

def binary_op(op, x, y, ret):
  if op == BinaryOps.ADD: code = "a+b"
  elif op == BinaryOps.SUB: code = "a-b"
  elif op == BinaryOps.MUL: code = "a*b"
  elif op == BinaryOps.DIV: code = "b/a"
  elif op == BinaryOps.POW: code = "pow(a,b)"
  elif op == BinaryOps.A: code = "a"
  elif op == BinaryOps.CMP: code = "1.0f*(a==b)"
  elif op == BinaryOps.EXPMUL: code = "exp(a) * b"
  elif op == BinaryOps.POW_D1: code = 'b * pow(a, b-1.0f)'
  elif op == BinaryOps.POW_D2: code = 'log(a) * pow(a, b)'
  elif op == BinaryOps.RELU_D: code = 'b * max(sign(a), (float)0.)'
  else: raise Exception(f"{op} isn't supported")

  shape_ret, dimlist, complist = binary_broadcast(x.shape, y.shape, True)
  assert tuple(shape_ret) == tuple(ret.shape)
  prod_list = np.array(dimlist, dtype=i32)[-1::-1].cumprod(dtype=i32)[-1::-1] # take cumprod from back to front
  prg, is_float4 = get_binop_prg(code, tuple(complist))
  kernel_size = ((roundup(prod_list[0])//4) if is_float4 else prod_list[0]) if len(dimlist) > 0 else 1
  prg.binop(cl_queue, [kernel_size], None, x.cl, y.cl, ret.cl, *dimlist, *(prod_list[1:]))
  return ret

def reduce_op(op, inp, ret, start="0.0"):
  if op == ReduceOps.SUM: code = "out += a"
  elif op == ReduceOps.MAX: code = "out = max(a,out)"
  elif op == ReduceOps.NEGSUM: code = "out -= a"
  else: raise Exception(f"{op} isn't supported")
  # TODO: this is insanely slow
  # NOTE: ret.shape can be (1,), it's mostly by luck that this works
  reduce = clbuild("reduce", """
  __kernel void reduce(__global const float *a_g, int sz, __global float *res_g, int prod, int n_dims,
                       __global const int *shape_x, __global const int *shape_ret) {
    int gid = get_global_id(0);

    float out = """+start+""";
    for (int x = 0; x < sz; x++) {
      int idx = 0;  // compute index into a_g
      int tprod = prod;
      int tsz = sz;
      for (int dim = 0; dim < n_dims; dim++) {
        idx *= shape_x[dim];
        if (shape_x[dim] == shape_ret[dim]) {   // dim from gid, don't reduce
          tprod /= shape_x[dim];
          idx += (gid / tprod) % shape_x[dim];
        } else {  // dim from x
          tsz /= shape_x[dim];
          idx += (x / tsz) % shape_x[dim];
        }
      }
      float a = a_g[idx];
      """+code+""";
    }
    res_g[gid] = out;
  }""")
  reduce([np.prod(ret.shape)], None, inp.cl,
    i32(np.prod(inp.shape)//np.prod(ret.shape)), ret.cl,
    i32(np.prod(ret.shape)), i32(len(ret.shape)),
    buffer_np(np.array(inp.shape, dtype=np.int32)),
    buffer_np(np.array(ret.shape, dtype=np.int32)))
  return ret


def perm_axis(inp, order, ret):
  perm = clbuild("perm", """
  __kernel void perm(__global const float *a_g, __global float *res_g, int n_axis,
                       __global const int *shape, __global const int *order) {
    int gid = get_global_id(0);
    int gi = gid;
    int idx = 0;
    for(int i = n_axis-1; i>-1; i--) {
      int stride = 1;
      for(int j=order[i]+1; j<n_axis; j++) stride *= shape[j];
      idx += (gi % shape[order[i]])*stride;
      gi /= shape[order[i]];
    }
    res_g[gid] = a_g[idx];
  }""")
  perm([np.prod(inp.shape)], None, inp.cl, ret.cl, i32(len(inp.shape)),
    buffer_np(np.array(inp.shape, dtype=np.int32)),
    buffer_np(np.array(order, dtype=np.int32)))
  return ret

# TODO: merge this with perm axis
def inner_slice(x, arg, ret):
  shift = [y[0] for y in arg]
  gslice = clbuild("gslice", """
  __kernel void gslice(__global const float *input, __global float *output, int prod, int n_dims,
                       __global const int *shape_x, __global const int *shape_ret,
                       __global const int *shift) {
    int gid = get_global_id(0);
    int iptr = 0;
    int zero = 1;
    for (int dim = 0; dim < n_dims; dim++) {
      prod /= shape_ret[dim];
      int sidx = (gid / prod) % shape_ret[dim] + shift[dim];
      zero &= (sidx >= 0 && sidx < shape_x[dim]);
      iptr = (iptr * shape_x[dim]) + sidx;
    }
    output[gid] = zero ? input[iptr] : 0.0;
  }""")
  gslice([np.prod(ret.shape)], None,
    x.cl, ret.cl, i32(np.prod(ret.shape)), i32(len(ret.shape)),
    buffer_np(np.array(x.shape, dtype=np.int32)),
    buffer_np(np.array(ret.shape, dtype=np.int32)),
    buffer_np(np.array(shift, dtype=np.int32)))
  return ret

# c = a@b
def matmul(a, b, c, transpose_a=False, transpose_b=False):
  cnt = np.prod(a.shape[0:-2]) if len(a.shape) > 2 else 1
  isize, msize, osize = i32(a.shape[-2]), i32(a.shape[-1]), i32(c.shape[-1])
  if transpose_a: isize,msize = msize,isize
  assert isize == c.shape[-2]
  assert (msize == b.shape[-1]) if transpose_b else (msize == b.shape[-2])
  assert (osize == b.shape[-2]) if transpose_b else (osize == b.shape[-1])
  
  matmul_prg = clbuild("matmul", """
  __kernel void matmul(
    __global const float *input, __global const float *weight, __global float *res,
    int isize, int is0, int is1, int msize, int ws0, int ws1, int osize
  ) {
    int stride = get_global_id(2);

    int X = get_global_id(0); // isize
    int Y = get_global_id(1); // osize

    float ret = 0.0;
    for (int x = 0; x < msize; x++) {
      ret += input[X * is0 + x * is1 + isize*msize*stride] * weight[Y * ws0 + x * ws1 + msize*osize*stride];
    }

    res[X * osize + Y + isize*osize*stride] = ret;
  }""")

  matmul_prg([isize, osize, cnt], None,
    a.cl, b.cl, c.cl,
    isize,
    msize if not transpose_a else i32(1), i32(1) if not transpose_a else isize,
    msize,
    i32(1) if not transpose_b else msize, osize if not transpose_b else i32(1),
    osize)
  return c


# TODO: combine any of these three?
def conv(x,w,ret,conv_args):
  H, W, groups, rcout, cin, oy, ox, iy, ix, ys, xs, bs = conv_args

  # input  = (bs, groups, cin, iy, ix)
  # weight = (groups, rcout, cin, H, W)
  # output = (bs, groups, rcout, oy, ox)
  conv_prg = clbuild("conv", """
  __kernel void conv(__global const float *input, __global const float *weight, __global float *output,
    int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs, int bs) {

    int B = get_global_id(0)/(groups*rcout);  // range 0-bs
    int g = (get_global_id(0)/rcout)%groups;
    int c = get_global_id(0) % rcout;

    int Y = get_global_id(1);  // range 0-oy
    int X = get_global_id(2);  // range 0-ox
    int IY = Y*ys;
    int IX = X*xs;

    float acc = 0.0;
    for (int ci = 0; ci < cin; ci++) {
      for (int y = IY; y < IY+H; y++) { for (int x = IX; x < IX+W; x++) {
        acc += input[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + y*ix + x] * \
          weight[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + (y-IY)*W + (x-IX)];
      } }
    }
    output[B*groups*rcout*oy*ox + g*rcout*oy*ox + c*oy*ox + Y*ox + X] = acc;
  }""")

  conv_prg([bs*groups*rcout, oy, ox], None, x.cl, w.cl, ret.cl, *[i32(x) for x in conv_args])
  return ret

# tensx = (bs, groups*cin, iy, ix)
# tensw = (groups*rcout, cin, H, W)
# ggg = (bs, groups*rout, oy, ox)

def convdw(x,grad_output,dw,conv_args):
  H, W, groups, rcout, cin, oy, ox, iy, ix, ys, xs, bs = conv_args
  convdw_prg = clbuild("convdw", """
  __kernel void convdw(__global const float *tensx, __global const float *ggg, __global float *dw,
    int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs, int bs) {

    int g = get_global_id(0)/(rcout*cin) ; // range 0-groups
    int c = (get_global_id(0)/(cin)) %rcout; // range 0-rcout
    int ci = get_global_id(0) % cin;        // range 0-cin
    int y = get_global_id(1);  // range 0-H
    int x = get_global_id(2);  // range 0-W

    float acc = 0.0;
    for (int Y = 0; Y < oy; Y++) { for (int X = 0; X < ox; X++) {
      for (int B = 0; B < bs; B++) {
        acc += ggg[B*groups*rcout*oy*ox + +g*rcout*oy*ox + c*oy*ox + Y*ox + X] * \
          tensx[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + (Y*ys+y)*ix + X*xs+x];
        }
    } }
    dw[get_global_id(0)*H*W + y*W + x] = acc;
  }""")
  convdw_prg([groups*rcout*cin, H, W], None, x.cl, grad_output.cl, dw.cl, *[i32(x) for x in conv_args])
  return dw

def convdx(w,grad_output,dx,conv_args):
  H, W, groups, rcout, cin, oy, ox, iy, ix, ys, xs, bs = conv_args
  convdx_prg = clbuild("convdx", """
  __kernel void convdx(__global const float *tensw, __global const float *ggg, __global float *dx,
    int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs, int bs) {

    int B = get_global_id(0);
    int g = get_global_id(1);
    int ci = get_global_id(2);

    for (int Y = 0; Y < iy; Y++) { for (int X = 0; X < ix; X++) {
      dx[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + Y*ix + X] = 0.0;
    } }

    for (int Y = 0; Y < oy; Y++) { for (int X = 0; X < ox; X++) {
      for (int y = 0; y < H; y++) { for (int x = 0; x < W; x++) {
        float acc = 0.0;
        for (int c = 0; c < rcout; c++) {
          acc += ggg[B*groups*rcout*oy*ox + g*rcout*oy*ox + c*oy*ox + Y*ox + X] * \
            tensw[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + y*W + x];
        }
        dx[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + (Y*ys+y)*ix + X*xs+x] += acc;
      } }
    } }
  }
  """)
  convdx_prg([bs, groups, cin], None, w.cl, grad_output.cl, dx.cl, *[i32(x) for x in conv_args])
  return dx

