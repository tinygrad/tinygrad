import numpy as np
from functools import lru_cache
from tinygrad.tensor import Function
from tinygrad.helpers import binary_broadcast

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

dev = cuda.Context.get_device()
MAX_THREADS_BLOCK = dev.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)

i32 = np.int32

class CudaBuffer:
  def __init__(self, shape, hostbuf=None):
    self.shape = tuple(shape)
    self.dtype = np.float32
    self.sz = int(np.prod(shape)*4)
    self.buf = cuda.mem_alloc(self.sz)

    if hostbuf is not None:
      if isinstance(hostbuf, CudaBuffer):
        self.buf = hostbuf.buf
      else:
        cuda.memcpy_htod(self.buf, hostbuf.flatten().astype(np.float32))

  @staticmethod
  def fromCPU(data):
    if data.dtype != np.float32:
      raise Exception('Only float32 is supported')
    return CudaBuffer(data.shape, data)

  def toCPU(self):
    ret = np.empty(self.shape).astype(np.float32)
    cuda.memcpy_dtoh(ret, self.buf)
    return ret

def buffer_new(shape, zero=False):
  return CudaBuffer(shape, hostbuf=None if not zero else np.zeros(shape, dtype=np.float32))

def buffer_np(x):
  x = np.array(x).astype(np.float32)
  buf = cuda.mem_alloc(int(np.prod(x.shape)*4))
  cuda.memcpy_htod(buf, x)
  return buf

def get_block_grid(shape=None, nelem=None):
  if shape is not None:
    nelem = int(np.prod(shape))
  block = (int([nelem, MAX_THREADS_BLOCK][int(nelem > MAX_THREADS_BLOCK)]), 1, 1)
  grid = (int(1+(nelem-1)//MAX_THREADS_BLOCK), 1)
  return block, grid

# ************* unary ops *************

def unary_op(code, x):
  block, grid = get_block_grid(x.shape)

  mod = SourceModule(f"""
  __global__ void unop(float *dest, float *a_g, int bufsz) {{
    const int i = blockDim.x*blockIdx.x+threadIdx.x;
    if (i < bufsz) {{
      float a = a_g[i];
      dest[i] = {code};
    }}
  }}
  """)
  unop = mod.get_function("unop")

  ret = buffer_new(x.shape)
  unop(ret.buf, x.buf, i32(np.prod(x.shape)), block=block, grid=grid)
  return ret

class ReLU(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return unary_op('max(a, (float)0.)', input)

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op('a * (b >= 0)', grad_output, input)

class Log(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return unary_op('log(a)', input)

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op('a / b', grad_output, input)

class Exp(Function):
  def forward(ctx, input):
    ret = unary_op('exp(a)', input)
    ctx.save_for_backward(ret)
    return ret

  def backward(ctx, grad_output):
    ret, = ctx.saved_tensors
    return binary_op('a * b', grad_output, ret)

# ************* reduce ops *************

def reduce_prg(x, axis, code, code2, start):
  rshape = x.shape[:1] if sum(x.shape[:axis+1]) == axis else list(x.shape[:axis]) + list(x.shape[axis+1:])
  stride = np.prod(x.shape[axis+1:], dtype=i32)
  bstride = np.prod(x.shape[axis:], dtype=i32) # stride to next "block"
  nsums = np.prod(rshape, dtype=i32)
  block, grid = get_block_grid(nelem=nsums)

  mod = SourceModule(f"""
  __global__ void prg(float *res, float *a_g, int stride, int bstride, int axis_dim, int nsums) {{
    const int d_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int idx = d_idx%stride;
    const int n = d_idx/stride;

    float out = {start};
    if (d_idx<nsums) {{
      for (int i=0; i<axis_dim ; i++) {{
        float a = a_g[idx+stride*i+n*bstride];
        {code};
      }}
      res[d_idx] = {code2};
    }}
  }}
  """)
  prg = mod.get_function("prg")

  ret = buffer_new(rshape)
  prg(ret.buf, x.buf, stride, bstride, i32(x.shape[axis]), nsums, block=block, grid=grid)
  return ret

def reduce_op(code, code2, ret, axis=None, start='0.0'):
  if axis is None:
    axis = list(range(len(ret.shape)))
    new_shape = (1,)
  else:
    new_shape = np.array(ret.shape)
    new_shape[axis if isinstance(axis, int) else list(axis)] = 1
    new_shape = tuple(new_shape)

  axis = sorted(axis)
  # reduces one axis at a time
  for i in range(len(axis)):
    ret = reduce_prg(ret, axis[i]-i, code, code2, start)

  ret.shape = new_shape
  return ret

class Sum(Function):
  def forward(ctx, input, axis=None):
    ctx.save_for_backward(input, axis)
    return reduce_op("out += a", "out", input, axis=axis)

  def backward(ctx, grad_output):
    input, _ = ctx.saved_tensors
    ret=binary_op('a+b', grad_output, buffer_new(input.shape, zero=True))
    return ret

class Max(Function):
  def forward(ctx, input, axis=None):
    ret = reduce_op("out = max(a,out)", "out", input, axis=axis, start="-INFINITY")
    ctx.save_for_backward(input, axis, ret)
    return ret

  def backward(ctx, grad_output):
    input, axis, ret = ctx.saved_tensors
    ret2 = binary_op("1.0*(a==b)", input, ret)
    div = reduce_op("out += a", "out+1e-10", ret2, axis=axis)
    ret3 = binary_op("a/b", ret2, div)
    return binary_op('a*b', ret3, grad_output)

# ************* binary ops *************

@lru_cache
def get_binop_prg(code, complist):
  ndims = len(complist)
  args = "".join([f", int d{i}" for i in range(ndims)] + [f", int p{i}" for i in range(ndims-1)])
  compute_idx_rets = "".join([f"\n    int idx_ret{i} = (gid0 / {f'p{i}' if i < ndims-1 else '1'}) % d{i};" for i in range(ndims)])

  idx_exprs = ["0", "0"] # [idx_x, idx_y]
  for i in range(ndims):
    for j in range(2):
      if complist[i][j]:
        idx_exprs[j] = f"idx_ret{i} + d{i}*({idx_exprs[j]})"

  mod = SourceModule(f"""
  __global__ void binop(float *res, float *a_g, float *b_g, int bufsz{args}) {{
    const int gid0 = blockIdx.x*blockDim.x + threadIdx.x;{compute_idx_rets}
    float a = a_g[{idx_exprs[0]}];
    float b = b_g[{idx_exprs[1]}];
    if (gid0 < bufsz) {{
      res[gid0] = {code};
    }}
  }}
  """)
  return mod

def binary_op(code, x, y):
  shape_ret, dimlist, complist = binary_broadcast(x.shape, y.shape)
  block, grid = get_block_grid(shape_ret)
  prod_list = np.array(dimlist, dtype=i32)[-1::-1].cumprod(dtype=i32)[-1::-1] # take cumprod from back to front

  prg = get_binop_prg(code, tuple(complist)).get_function('binop')

  ret = buffer_new(shape_ret, zero=True)
  prg(ret.buf, x.buf, y.buf, i32(np.prod(shape_ret)), *dimlist, *(prod_list[1:]), block=block, grid=grid)
  return ret

def unbroadcast(out, in_sh):
  sum_axis = [i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1] if in_sh != (1,) else None
  return reduce_op("out += a", "out", out, sum_axis)

class Add(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return binary_op('a+b', x, y)

  def backward(ctx, grad_output):
    grad_x, grad_y = grad_output, grad_output
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(grad_x, shape_x), unbroadcast(grad_y, shape_y)

class Sub(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return binary_op('a-b', x, y)

  def backward(ctx, grad_output):
    grad_x, grad_y = grad_output, unary_op('-a', grad_output)
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(grad_x, shape_x), unbroadcast(grad_y, shape_y)

class Mul(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op('a*b', x, y)

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    grad_x = binary_op('a*b', y, grad_output)
    grad_y = binary_op('a*b', x, grad_output)
    return unbroadcast(grad_x, x.shape), unbroadcast(grad_y, y.shape)

class Pow(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op('pow(a,b)', x, y)

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    grad_x = binary_op('a*b', grad_output,
                      binary_op('b * (pow((float)a, (float)(b-1.0)))', x, y))
    grad_y = binary_op('a*b', grad_output,
                      binary_op('pow(a, (float)b) * log(a);', x, y))
    return unbroadcast(grad_x, x.shape), unbroadcast(grad_y, y.shape)

# ************* movement ops *************

class Reshape(Function):
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    shape = tuple(-np.prod(x.shape) // np.prod(shape) if s == -1 else s for s in shape)
    r = CudaBuffer(shape, hostbuf=x)   # NOTE: this is not a copy
    assert np.prod(x.shape) == np.prod(r.shape)
    return r

  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return CudaBuffer(in_shape, hostbuf=grad_output)

def perm_axis(inp, order):
  osize = np.array(inp.shape)[list(order)]
  ret = buffer_new(osize)

  nthr = int(np.prod(osize))
  block, grid = get_block_grid(osize)

  perm = SourceModule("""
  __global__ void perm(float *a_g, float *res_g, int n_axis,
                       float *shape, float *order, int bufsz) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int gi = gid;
    int idx = 0;
    if (gid < bufsz) {
      for(int i = n_axis-1; i>-1; i--) {
        int stride = 1;
        for(int j=(int)order[i]+1; j<n_axis; j++) stride *= (int)shape[j];
        idx += (gi % (int)shape[(int)order[i]])*stride;
        gi /= (int)shape[(int)order[i]];
      }
      res_g[gid] = a_g[idx];
    }
  }""").get_function("perm")

  perm(inp.buf, ret.buf, i32(len(osize)),
    buffer_np(np.array(inp.shape, dtype=np.float32)),
    buffer_np(np.array(order, dtype=np.float32)), i32(nthr),
    block=block, grid=grid)

  return ret

class Transpose(Function):
  def forward(ctx, x, order=(1,0)):
    ctx.save_for_backward(order)
    return perm_axis(x, order)

  def backward(ctx, grad_output):
    return perm_axis(grad_output, np.argsort(ctx.order))

# TODO: merge this with perm axis
def inner_slice(x, arg):
  shift = [y[0] for y in arg]
  oshape = [y[1]-y[0] for y in arg]
  ret = buffer_new(oshape)

  nthr = int(np.prod(oshape))
  block, grid = get_block_grid(oshape)

  gslice = SourceModule("""
  __global__ void gslice(float *input, float *output, int prod, int n_dims,
                         float *shape_x, float *shape_ret, float *shift, int bufsz) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid < bufsz) {
      int iptr = 0;
      int zero = 1;
      for (int dim = 0; dim < n_dims; dim++) {
        prod /= (int)shape_ret[dim];
        int sidx = (gid / prod) % (int)shape_ret[dim] + (int)shift[dim];
        zero &= (sidx >= 0 && sidx < (int)shape_x[dim]);
        iptr = (iptr * (int)shape_x[dim]) + sidx;
      }
      output[gid] = zero ? input[iptr] : 0.0;
    }
  }""").get_function('gslice')

  gslice(x.buf, ret.buf, i32(np.prod(ret.shape)), i32(len(ret.shape)),
    buffer_np(np.array(x.shape, dtype=np.int32)),
    buffer_np(np.array(ret.shape, dtype=np.int32)),
    buffer_np(np.array(shift, dtype=np.int32)), i32(nthr),
    block=block, grid=grid)

  return ret

class Slice(Function):
  def forward(ctx, x, arg=None):
    ctx.save_for_backward(x.shape)
    return inner_slice(x, arg)

  def backward(ctx, grad_output):
    shape, = ctx.saved_tensors
    narg = [(0-p[0], grad_output.shape[i]+(shape[i]-p[1])) for i,p in enumerate(ctx.arg)]
    return inner_slice(grad_output, narg)

# ************* processing ops *************

class Matmul(Function):
  def forward(ctx, input, weight):
    assert input.shape[-1] == weight.shape[-2]
    cnt = np.prod(input.shape[0:-2]) if len(input.shape) > 2 else 1
    isize, msize, osize = i32(input.shape[-2]), i32(input.shape[-1]), i32(weight.shape[-1])
    ret = buffer_new(list(input.shape[0:-2])+[isize, osize])

    nthr = int(np.prod(ret.shape))
    block, grid = get_block_grid(nelem=nthr)

    matmul = SourceModule("""
    __global__ void matmul(float *input, float *weight, float *res,
                           int isize, int is0, int is1, int msize,
                           int ws0, int ws1, int osize, int bufsz) {
      int gid = blockIdx.x*blockDim.x + threadIdx.x;

      int stride = gid/(osize*isize);

      int X = gid%isize; // isize
      int Y = (gid/isize)%osize; // osize

      int ind = X * osize + Y + isize*osize*stride;
      if (ind < bufsz) {
        float ret = 0.0;
        for (int x = 0; x < msize; x++) {
          ret += input[X * is0 + x * is1 + isize*msize*stride] *
            weight[Y * ws0 + x * ws1 + msize*osize*stride];
        }
        res[ind] = ret;
      }
    }""").get_function('matmul')

    ctx.save_for_backward(input, weight, matmul, cnt)

    # (isize,msize) x (msize,osize) = (isize,osize)
    matmul(input.buf, weight.buf, ret.buf, i32(isize),
      i32(msize), i32(1), i32(msize), i32(1), i32(osize),
      i32(osize), i32(nthr), block=block, grid=grid)

    return ret

  def backward(ctx, grad_output):
    input, weight, matmul, cnt = ctx.saved_tensors
    isize, msize, osize = i32(input.shape[-2]), i32(input.shape[-1]), i32(weight.shape[-1])

    grad_input = buffer_new(input.shape)
    grad_weight = buffer_new(weight.shape)

    nthr = int(np.prod(grad_input.shape))
    block, grid = get_block_grid(nelem=nthr)

    # (isize,osize) x (msize,osize) = (isize,msize)
    matmul(grad_output.buf, weight.buf, grad_input.buf, i32(isize),
      i32(osize), i32(1), i32(osize), i32(osize), i32(1),
      i32(msize), i32(nthr), block=block, grid=grid)

    nthr = int(np.prod(grad_weight.shape))
    block, grid = get_block_grid(nelem=nthr)

    # (isize,msize) x (isize,osize) = (msize,osize)
    matmul(input.buf, grad_output.buf, grad_weight.buf, i32(msize), i32(1),
      i32(msize), i32(isize), i32(1), i32(osize),
      i32(osize), i32(nthr), block=block, grid=grid)

    return grad_input, grad_weight

class Conv2D(Function):
  def forward(ctx, x, w, stride=1, groups=1):
    if isinstance(ctx.stride, int): ctx.stride = (ctx.stride, ctx.stride)
    cout,cin,H,W = w.shape
    ys,xs = ctx.stride
    bs,cin_,iy,ix = x.shape
    oy,ox = (iy-(H-ys))//ys, (ix-(W-xs))//xs
    if cin*ctx.groups != cin_:
      raise Exception(f"Input Tensor shape {x.shape} does not match the shape of the weights {w.shape}. ({cin*ctx.groups} vs. {cin_})")
    assert cout % ctx.groups == 0
    rcout = cout//ctx.groups

    ctx.save_for_backward(x,w)

    # output buffer
    ret = buffer_new((bs, cout, oy, ox))

    nthr = int(np.prod(ret.shape))
    block, grid = get_block_grid(nelem=nthr)

    conv = SourceModule("""
    __global__ void conv(float *input, float *weight, float *output,
      int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs, int bs, int bufsz) {

      const int gid = blockIdx.x*blockDim.x + threadIdx.x;

      int B = gid/(groups*rcout*oy*ox);  // range 0-bs
      int g = (gid/(rcout*oy*ox))%groups;
      int c = (gid/(oy*ox))%rcout;

      int Y = gid%oy;  // range 0-oy
      int X = (gid/oy)%ox;  // range 0-ox
      int IY = Y*ys;
      int IX = X*xs;

      int ind = B*groups*rcout*oy*ox + g*rcout*oy*ox + c*oy*ox + Y*ox + X;
      if (ind < bufsz) {
        float acc = 0.0;
        for (int ci = 0; ci < cin; ci++) {
          for (int y = IY; y < IY+H; y++) {
            for (int x = IX; x < IX+W; x++) {
              acc += input[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + y*ix + x] *
                weight[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + (y-IY)*W + (x-IX)];
            }
          }
        }
        output[ind] = acc;
      }
    }""").get_function('conv')


    conv(x.buf, w.buf, ret.buf,
      i32(H), i32(W), i32(groups), i32(rcout), i32(cin),
      i32(oy), i32(ox), i32(iy), i32(ix), i32(ys), i32(xs), i32(bs), i32(nthr),
      block=block, grid=grid)

    return ret

  def backward(ctx, grad_output):
    bs,_,oy,ox = grad_output.shape
    x, w = ctx.saved_tensors
    cout,cin,H,W = w.shape
    ys,xs = ctx.stride
    bs,cin_,iy,ix = x.shape
    oy,ox = (iy-(H-ys))//ys, (ix-(W-xs))//xs
    assert cin*ctx.groups == cin_
    assert cout % ctx.groups == 0
    rcout = cout//ctx.groups

    convw = SourceModule("""
    __global__ void convw(float *tensx, float *ggg, float *dw,
      int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs, int bs, int bufsz) {

      const int gid = blockIdx.x*blockDim.x + threadIdx.x;

      int g = gid/(rcout*cin*H*W); // range 0-groups
      int c = (gid/(cin*H*W))%rcout; // range 0-rcout
      int ci = (gid/(H*W))%cin;        // range 0-cin
      int y = gid%H;  // range 0-H
      int x = (gid/H)%W;  // range 0-W

      int ind = (gid/(H*W))*H*W + y*W + x;
      if (ind < bufsz) {
        float acc = 0.0;
        for (int Y = 0; Y < oy; Y++) {
          for (int X = 0; X < ox; X++) {
            for (int B = 0; B < bs; B++) {
              acc += ggg[B*groups*rcout*oy*ox + +g*rcout*oy*ox + c*oy*ox + Y*ox + X] *
                tensx[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + (Y*ys+y)*ix + X*xs+x];
            }
          }
        }
        dw[ind] = acc;
      }
    }""").get_function('convw')

    convx = SourceModule("""
    __global__ void convx(float *tensw, float *ggg, float *dx,
      int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs, int bs, int bufsz) {

      const int gid = blockIdx.x*blockDim.x + threadIdx.x;

      int B = gid/(groups*cin);
      int g = gid%groups;
      int ci = (gid/groups)%cin;

      for (int Y = 0; Y < oy; Y++) {
        for (int X = 0; X < ox; X++) {
          for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
              int ind = B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + (Y*ys+y)*ix + X*xs+x;
              if (ind < bufsz) {
                float acc = 0.0;
                for (int c = 0; c < rcout; c++) {
                  acc += ggg[B*groups*rcout*oy*ox + g*rcout*oy*ox + c*oy*ox + Y*ox + X] *
                    tensw[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + y*W + x];
                }
                dx[ind] += acc;
              }
            }
          }
        }
      }
    }
    """).get_function('convx')

    conv_args = i32(H), i32(W), i32(ctx.groups), i32(rcout), i32(cin), i32(oy), i32(ox), i32(iy), i32(ix), i32(ys), i32(xs), i32(bs)

    dw = buffer_new(w.shape)
    nthr = int(ctx.groups*rcout*cin*H*W)
    block, grid = get_block_grid(nelem=nthr)
    convw(x.buf, grad_output.buf, dw.buf, *conv_args, i32(np.prod(w.shape)), block=block, grid=grid)

    dx = buffer_new(x.shape, True)
    nthr = int(bs*ctx.groups*cin)
    block, grid = get_block_grid(nelem=nthr)
    convx(w.buf, grad_output.buf, dx.buf, *conv_args, i32(np.prod(x.shape)), block=block, grid=grid)

    return dx, dw
