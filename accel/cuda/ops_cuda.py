import numpy as np
from functools import lru_cache
from tinygrad.tensor import Function
from tinygrad.helpers import binary_broadcast

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceData

from rich import print

dev = cuda.Context.get_device()
devdata = DeviceData(dev)

MAX_THREADS_BLOCK = dev.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)

class CudaBuffer:
  def __init__(self, shape, hostbuf=None):
    self.shape = shape
    self.sz = int(np.prod(shape)*4)

    self.buf = cuda.mem_alloc(self.sz)
    if hostbuf is not None:
      if isinstance(hostbuf, CudaBuffer):
        self.buf = hostbuf.buf
      else:
        cuda.memcpy_htod(self.buf, hostbuf)

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

  print(idx_exprs, compute_idx_rets)
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
  prod_list = np.array(dimlist, dtype=np.int32)[-1::-1].cumprod(dtype=np.int32)[1::-1] # take cumprod from back to front

  prg = get_binop_prg(code, tuple(complist)).get_function('binop')
  ret = buffer_new(shape_ret, zero=True)
  block,grid = get_block_grid(shape_ret)
  print("dimlist", dimlist)
  print("prodlist", prod_list[1:], prod_list)
  prg(ret.buf, x.buf, y.buf, *dimlist, *(prod_list[1:]), block=block, grid=grid)

  return ret

def unbroadcast(ctx, out, in_sh):
  sum_axis = [i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1] if in_sh != (1,) else None
  return reduce_op(ctx, "out += a", "out", out, sum_axis)


def sum_prg(x,axis):
  rshape = x.shape[:1] if sum(x.shape[:axis+1]) == axis else x.shape[:axis] + x.shape[axis+1:]
  stride = np.prod(x.shape[axis+1:], dtype=np.int32)
  jmplen = np.prod(x.shape[axis:], dtype=np.int32) # distance to next "block"
  nsums = np.prod(rshape, dtype=np.int32)

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
  import time
  s = time.time()
  prg(ret.buf, x.buf, stride, jmplen, nsums, np.int32(x.shape[axis]), block=block, grid=grid)
  print("real time: ", time.time()-s)
  return ret

def sum_op(ret, axis):
    if isinstance(axis,int): axis = [axis]
    axis = sorted(axis)
    for i in range(len(axis)):
      ls = list(map(lambda x: x-1, axis[i+1:]))
      axis = axis[:i+1]+ls
    for ax in axis:
      ret = sum_prg(ret, ax)
    return ret

def reduce_op(code, code2, x, axis=None, start="0.0"):
  axis = 3 # tmp
  print(ret.toCPU())
  return ret

class Add(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x.shape, y.shape)
    return binary_op('a+b', x, y)

  def backward(ctx, grad_output):
    grad_x, grad_y = grad_output, grad_output
    shape_x, shape_y = ctx.saved_tensors
    return unbroadcast(ctx, grad_x, shape_x), unbroadcast(ctx, grad_y, shape_y)

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
    return unary_op(ctx, 'log(a)', input)

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op(ctx, 'a / b', grad_output, input)

class Exp(Function):
  def forward(ctx, input):
    ret = unary_op(ctx, 'exp(a)', input)
    ctx.save_for_backward(ret)
    return ret

  def backward(ctx, grad_output):
    ret, = ctx.saved_tensors
    return binary_op(ctx, 'a * b', grad_output, ret)

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

if __name__ == "__main__":

  from tinygrad.tensor import Tensor, Device

  n = 5
  test = np.arange(3*n**2)
  test = test.reshape(3,n,n)
  test = test.astype(np.float32)

  r1 = Tensor(test, device=Device.CUDA)
  r2 = Tensor(np.arange(50**2).reshape(50,50).astype(np.float32), device=Device.CUDA)
  #b1 = CudaBuffer((50000,5000), np.arange(5000*50000).astype(np.float32))

  #r3 = r1 + r2

  #print(r1.data.toCPU())
  #print(r2.data.toCPU())
  #print(r3.data.toCPU())

  #print('@@@@@')
  #r2 = r1.relu()
  #print(r2.data.toCPU())
  #r2 = Tensor(np.ones((3,n,n), dtype=np.float32), device=Device.CUDA)
  #r2 = Tensor(np.arange(2*2*2*2).reshape(2,2,2,2).astype(np.float32), device=Device.CUDA)

  #print('@@@@@')

  import time
  s = time.time()
  r3 = r2.sum(axis=0)
  #r3 = sum_op(r2, 0)
  #r3 = sum_op(b1, 1)
  print(time.time()-s)
  print(r3.data.toCPU())
  #r2.backward()
