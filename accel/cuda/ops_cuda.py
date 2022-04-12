import numpy as np
from functools import lru_cache
from tinygrad.ops.ops_cpu import CPUBuffer
from tinygrad.tensor import Function
from tinygrad.helpers import binary_broadcast

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceData
import pycuda.autoinit

from rich import print
import time
import gc

def init():
  import pycuda.autoinit
  dev = cuda.Context.get_device()
  devdata = DeviceData(dev)
  return dev.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)

MAX_THREADS_BLOCK = init()

i32 = np.int32

class CudaBuffer:
  def __init__(self, shape, hostbuf=None):
    gc.collect()
    self.shape = shape
    self.dtype = np.float32
    self.sz = int(np.prod(shape)*4)
    self.buf = cuda.mem_alloc(self.sz)

    if hostbuf is not None:
      if isinstance(hostbuf, CudaBuffer):
        self.buf = hostbuf.buf
      elif isinstance(hostbuf, CPUBuffer):
        cuda.memcpy_htod(self.buf, hostbuf.astype(np.float32))
      else:
        cuda.memcpy_htod(self.buf, hostbuf.astype(np.float32))

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

def get_block_grid(shape):
  nelem = int(np.prod(shape))
  block = (MAX_THREADS_BLOCK if nelem>MAX_THREADS_BLOCK else nelem, 1, 1)
  d,r = divmod(nelem, block[0])
  grid = (d+(r>0), 1)
  return block, grid

def unary_op(code, x):
  block,grid = get_block_grid(x.shape)
  ret = buffer_new(x.shape)
  mod = SourceModule(f"""
  __global__ void unop(float *dest, float *a_g) {{
    const int i = blockDim.x*blockIdx.x+threadIdx.x;
    float a = a_g[i];
    dest[i] = {code};
  }}
  """)
  unop = mod.get_function("unop")
  unop(ret.buf, x.buf, block=block, grid=grid)
  return ret

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
  __global__ void binop(float *res, float *a_g, float *b_g{args}) {{
    const int gid0 = blockIdx.x*blockDim.x + threadIdx.x;{compute_idx_rets}
    float a = a_g[{idx_exprs[0]}];
    float b = b_g[{idx_exprs[1]}];
    res[gid0] = {code};
  }}
  """)
  return mod

def binary_op(code, x, y):
  shape_ret, dimlist, complist = binary_broadcast(x.shape, y.shape)
  prod_list = np.array(dimlist, dtype=i32)[-1::-1].cumprod(dtype=i32)[1::-1] # take cumprod from back to front

  prg = get_binop_prg(code, tuple(complist)).get_function('binop')
  ret = buffer_new(shape_ret, zero=True)
  block,grid = get_block_grid(shape_ret)
  prg(ret.buf, x.buf, y.buf, *dimlist, *(prod_list[1:]), block=block, grid=grid)

  return ret

def sum_prg(x,axis):
  rshape = x.shape[:1] if sum(x.shape[:axis+1]) == axis else list(x.shape[:axis]) + list(x.shape[axis+1:])
  stride = np.prod(x.shape[axis+1:], dtype=i32)
  jmplen = np.prod(x.shape[axis:], dtype=i32) # distance to next "block"
  nsums = np.prod(rshape, dtype=i32)

  # very costly
  ret = buffer_new(rshape, True)

  block = (int(MAX_THREADS_BLOCK if nsums>MAX_THREADS_BLOCK else nsums), 1, 1)
  grid = (int(nsums//MAX_THREADS_BLOCK + (nsums%MAX_THREADS_BLOCK>0)), 1)

  mod = SourceModule(f"""
  __global__ void sum(float *res, float *a_g, int stride, int jmplen, int nsums, int axis_dim) {{
    const int d_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int idx = d_idx%stride;
    const int n = d_idx/stride;

    float sum = 0;
    if (d_idx<nsums) {{
      for (int i=0; i<axis_dim ; i++) {{
        sum += a_g[idx+stride*i+n*jmplen];
      }}
      res[d_idx] = sum;
    }}
  }}
  """)

  prg = mod.get_function("sum")
  prg(ret.buf, x.buf, stride, jmplen, nsums, i32(x.shape[axis]), block=block, grid=grid)
  return ret

def sum_op(ret, axis):
  if isinstance(axis,int): axis = [axis]
  axis = sorted(axis)
  for i in range(len(axis)):
    print( axis[i]-i)
    ret = sum_prg(ret, axis[i]-i)
  return ret

def reduce_op(code, code2, inp, axis=None, start="0.0"):
  if axis is None:
    # full reduce
    osize = [1]*len(inp.shape)
  else:
    osize = np.array(inp.shape)
    osize[axis if isinstance(axis, int) else list(axis)] = 1
  ret = buffer_new(osize)
  if axis is None:
    ret.shape = (1,)

  block,grid = get_block_grid(ret.shape)

  # TODO: this is insanely slow
  reduce = SourceModule(f"""
  __global__ void reduce(float *a_g, float *res_g, int sz, int prod, int n_dims, float *shape_x, float *shape_ret) {{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    float out = {start};
    for (int x = 0; x < sz; x++) {{
      int idx = 0;  // compute index into a_g
      int tprod = prod;
      int tsz = sz;
      for (int dim = 0; dim < n_dims; dim++) {{
        int tshapex = (int) shape_x[dim];
        idx *= tshapex;
        if (tshapex == (int) shape_ret[dim]) {{   // dim from gid, don't reduce
          tprod /= tshapex;
          idx += (gid / tprod) % tshapex;
        }} else {{  // dim from x
          tsz /= tshapex;
          idx += (x / tsz) % tshapex;
        }}
      }}
      float a = a_g[idx];
      {code};
    }}
    res_g[gid] = {code2};
  }}""").get_function("reduce")

  reduce(
    inp.buf, ret.buf,
    i32(np.prod(inp.shape)//np.prod(osize)),
    i32(np.prod(osize)), i32(len(osize)),
    buffer_np(np.array(inp.shape, dtype=np.float32)),
    buffer_np(np.array(osize, dtype=np.float32)),
    block=block, grid=grid)
  return ret

def unbroadcast(out, in_sh):
  axis = [i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1] if in_sh != (1,) else None
  #return reduce_op("out += a", "out", out, sum_axis)
  if axis is None:
    # full reduce
    axis = list(range(len(in_sh)))

  print(out, axis)
  return sum_op(out, axis)

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

class Sum(Function):
  def forward(ctx, input, axis=None):
    ctx.save_for_backward(input)
    return sum_op(input, axis)

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op('a * (b >= 0)', grad_output, input)

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
  block = ([nthr, MAX_THREADS_BLOCK][nthr > MAX_THREADS_BLOCK], 1, 1)
  grid = (1+(nthr-1)//MAX_THREADS_BLOCK, 1)

  perm = SourceModule(f"""
  __global__ void perm(float *a_g, float *res_g, int n_axis,
                       float *shape, float *order) {{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int gi = gid;
    int idx = 0;
    for(int i = n_axis-1; i>-1; i--) {{
      int stride = 1;
      for(int j=order[i]+1; j<n_axis; j++) stride *= shape[j];
      idx += (gi % shape[order[i]])*stride;
      gi /= shape[order[i]];
    }}
    res_g[gid] = a_g[idx];
    }}""").get_function("perm")

  perm(inp.buf, ret.buf, i32(len(osize)),
    buffer_np(np.array(inp.shape, dtype=np.float32)),
    buffer_np(np.array(order, dtype=np.float32)),
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

  nthr = int(np.prod(ret.shape))
  block = ([nthr, MAX_THREADS_BLOCK][nthr > MAX_THREADS_BLOCK], 1, 1)
  grid = (1+(nthr-1)//MAX_THREADS_BLOCK, 1)
 
  gslice = SourceModule("""
  __global__ void gslice(float *input, float *output, int prod, int n_dims,
                         float *shape_x, float *shape_ret, float *shift) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int iptr = 0;
    int zero = 1;
    for (int dim = 0; dim < n_dims; dim++) {
      prod /= shape_ret[dim];
      int sidx = (gid / prod) % shape_ret[dim] + shift[dim];
      zero &= (sidx >= 0 && sidx < shape_x[dim]);
      iptr = (iptr * shape_x[dim]) + sidx;
    }
    output[gid] = zero ? input[iptr] : 0.0;
  }""").get_function('gslice')

  gslice(x.cl, ret.cl, i32(np.prod(ret.shape)), i32(len(ret.shape)),
    buffer_np(np.array(x.shape, dtype=np.int32)),
    buffer_np(np.array(ret.shape, dtype=np.int32)),
    buffer_np(np.array(shift, dtype=np.int32)),
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

    nthr = int(cnt*isize*osize)
    block = ([nthr, MAX_THREADS_BLOCK][nthr > MAX_THREADS_BLOCK], 1, 1)
    grid = (1+(nthr-1)//MAX_THREADS_BLOCK, 1)

    matmul = SourceModule(f"""
    __global__ void matmul(float *input, float *weight, float *res,
                            int isize, int is0, int msize,
                            int ws1, int osize) {{
      int gid = blockIdx.x*blockDim.x + threadIdx.x;

      int stride = gid/(osize*isize);

      int X = gid%isize; // isize
      int Y = (gid/isize)%osize; // osize

      float ret = 0.0;
      for (int x = 0; x < msize; x++) {{
        ret += input[X * msize + x + isize*msize*stride] * weight[Y + x * osize + osize*msize*stride];
      }}

      res[X * osize + Y + isize*osize*stride] = ret;
    }}""").get_function('matmul')
    
    ctx.save_for_backward(input, weight, matmul, cnt)

    # (isize,msize) x (msize,osize) = (isize,osize)
    matmul(input.buf, weight.buf, ret.buf, i32(isize),
      i32(msize), i32(msize), i32(osize), i32(osize), block=block, grid=grid)
    return ret

  def backward(ctx, grad_output):
    input, weight, matmul, cnt = ctx.saved_tensors
    isize, msize, osize = i32(input.shape[-2]), i32(input.shape[-1]), i32(weight.shape[-1])

    grad_input = buffer_new(input.shape)
    grad_weight = buffer_new(weight.shape)

    nthr = int(cnt*isize*osize)
    block = ([nthr, MAX_THREADS_BLOCK][nthr > MAX_THREADS_BLOCK], 1, 1)
    grid = (1+(nthr-1)//MAX_THREADS_BLOCK, 1)

    # (isize,osize) x (msize,osize) = (isize,msize)
    matmul(grad_output.buf, weight.buf, grad_input.buf, isize,
      osize, osize, osize, msize, block=block, grid=grid)

    # (isize,msize) x (isize,osize) = (msize,osize)
    matmul(input.buf, grad_output.buf, grad_weight.buf, msize,
      msize, isize, osize, osize, block=block, grid=grid)

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

    nthr = int(bs*groups*rcout, oy, ox)
    block = ([nthr, MAX_THREADS_BLOCK][nthr > MAX_THREADS_BLOCK], 1, 1)
    grid = (1+(nthr-1)//MAX_THREADS_BLOCK, 1)

    # input  = (bs, groups, cin, iy, ix)
    # weight = (groups, rcout, cin, H, W)
    # output = (bs, groups, rcout, oy, ox)

    conv = SourceModule(f"""
    __kernel void conv(__global const float *input, __global const float *weight, __global float *output,
      int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs) {{

      int B = get_global_id(0)/(groups*rcout);  // range 0-bs
      int g = (get_global_id(0)/rcout)%groups;
      int c = get_global_id(0) % rcout;

      int Y = get_global_id(1);  // range 0-oy
      int X = get_global_id(2);  // range 0-ox
      int IY = Y*ys;
      int IX = X*xs;

      float acc = 0.0;
      for (int ci = 0; ci < cin; ci++) {{
        for (int y = IY; y < IY+H; y++) {{
          for (int x = IX; x < IX+W; x++) {{
            acc += input[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + y*ix + x] * \
              weight[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + (y-IY)*W + (x-IX)];
          }}
        }}
      }}
      output[B*groups*rcout*oy*ox + g*rcout*oy*ox + c*oy*ox + Y*ox + X] = acc;
    }}""").get_function('conv')

    conv(x.buf, w.buf, ret.buf,
      i32(H), i32(W), i32(groups), i32(rcout), i32(cin),
      i32(oy), i32(ox), i32(iy), i32(ix), i32(ys), i32(xs),
      block=block, grid=grid
    )
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

    dx = buffer_new((bs, cin_, iy, ix), zero=True)
    dw = buffer_new((cout, cin, H, W))

    # tensx = (bs, groups*cin, iy, ix)
    # tensw = (groups*rcout, cin, H, W)
    # ggg = (bs, groups*rout, oy, ox)

    convw = SourceModule("""
    __kernel void convw(__global const float *tensx, __global const float *ggg, __global float *dw,
      int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs, int bs) {

      int g = get_global_id(0)/(rcout*cin) ; // range 0-groups
      int c = (get_global_id(0)/(cin)) %rcout; // range 0-rcout
      int ci = get_global_id(0) % cin;        // range 0-cin
      int y = get_global_id(1);  // range 0-H
      int x = get_global_id(2);  // range 0-W

      float acc = 0.0;
      for (int Y = 0; Y < oy; Y++) {
        for (int X = 0; X < ox; X++) {
          for (int B = 0; B < bs; B++) {
            acc += ggg[B*groups*rcout*oy*ox + +g*rcout*oy*ox + c*oy*ox + Y*ox + X] * \
              tensx[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + (Y*ys+y)*ix + X*xs+x];
          }
        }
      }
      dw[get_global_id(0)*H*W + y*W + x] = acc;
    }""")
    convx = SourceModule("""
    __kernel void convx(__global const float *tensw, __global const float *ggg, __global float *dx,
      int H, int W, int groups, int rcout, int cin, int oy, int ox, int iy, int ix, int ys, int xs, int bs) {

      int B = get_global_id(0);
      int g = get_global_id(1);
      int ci = get_global_id(2);

      for (int Y = 0; Y < oy; Y++) {
        for (int X = 0; X < ox; X++) {
          for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
              float acc = 0.0;
              for (int c = 0; c < rcout; c++) {
                acc += ggg[B*groups*rcout*oy*ox + g*rcout*oy*ox + c*oy*ox + Y*ox + X] * \
                  tensw[g*rcout*cin*H*W + c*cin*H*W + ci*H*W + y*W + x];
              }
              dx[B*groups*cin*iy*ix + g*cin*iy*ix + ci*iy*ix + (Y*ys+y)*ix + X*xs+x] += acc;
            }
          }
        }
      }
    }
    """)

    conv_args = i32(H), i32(W), i32(ctx.groups), i32(rcout), i32(cin), i32(oy), i32(ox), i32(iy), i32(ix), i32(ys), i32(xs), i32(bs)
    convw(cl_queue, [ctx.groups*rcout*cin, H, W], None, x.cl, grad_output.cl, dw.cl, *conv_args)
    convx(cl_queue, [bs, ctx.groups, cin], None, w.cl, grad_output.cl, dx.cl, *conv_args)
    return dx, dw

if __name__ == "__main__":

  from tinygrad.tensor import Tensor, Device

  n = 5
  test = np.arange(3*n**2)
  test = test.reshape(3,n,n)
  test = test.astype(np.float32)

  #r1 = Tensor(test, device=Device.CUDA)
  #r2 = Tensor(np.arange(2*50000*5000).reshape(2,50000,5000).astype(np.float32), device=Device.CUDA)
  M = np.arange(2*2*3*50*40).reshape(2,2,3,50,40).astype(np.float32)
#  N = np.arange(3*10*400).reshape(3,400,10).astype(np.float32)
  r2 = Tensor(M, device=Device.CUDA)
 # r22 = Tensor(N, device=Device.CUDA)
  #r22 = CudaBuffer((2,50000,5000),np.arange(2*50000*5000).reshape(2,50000,5000).astype(np.float32))



  r3 = r2.sum(axis=(1,3,4,2)).data.toCPU()
  l = M.sum(axis=(1,4,3,2))
  print(r3, l, np.allclose(r3,l))
  exit()
  r3= r2.matmul(r22)
  r3 = r3.data.toCPU()
  print(r3)
  print()
  print('fasit')
  a=(M@N).astype(np.float32)
  print(a)
  print(np.allclose(a,r3))
  exit()
  binary_op('a+b', r22, r22)
  #print(r1.data.toCPU())
  print("steg1")
  import time
  s = time.time()
  a = sum_op(r22,0)
  #$a = r2.sum(axis=1)
  print(time.time()-s)
  print(a.toCPU())

  exit()
  r2 = r1+r1
  print(r2.data.toCPU())
  print("steg2")
  r3 = r2.sum()
  print(r3.data.toCPU())

  r3.backward()

  print(r3.data.toCPU())
  print("grad")
  print(r3.grad.data.toCPU())
  print(r1.grad.data.toCPU())

  #b1 = CudaBuffer((50000,5000), np.arange(5000*50000).astype(np.float32))

  #r3 = r1 + r2

  #print(r1.data.toCPU())
  #print(r2.data.toCPU())
  #print(r3.data.toCPU())

  #print('@@@@@')
  #r2 = r1.relu()
  #print(r2.data.toCPU())
  #r2 = Tensor(np.ones((4,n,n), dtype=np.float32), device=Device.CUDA)
  #r2 = Tensor(np.arange(2*2*2*2).reshape(2,2,2,2).astype(np.float32), device=Device.CUDA)

  #print('@@@@@')

  #import time
