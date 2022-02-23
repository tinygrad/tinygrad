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
#MAX_THREADS_MULTIPROCESSOR = dev.get_attribute(cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR)

class CudaBuffer:
  def __init__(self, shape, hostbuf=None):
    self.shape = shape
    self.sz = int(np.prod(shape)*4)

    self.buf = cuda.mem_alloc(self.sz)
    if hostbuf is not None:
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
  if len(shape) == 1:
    nelem = int(np.prod(shape))
    block = (MAX_THREADS_BLOCK if nelem>MAX_THREADS_BLOCK else nelem, 1, 1)
    d,r = divmod(shape[0], block[0])
    grid = (d+(r>0), 1)
  elif len(shape) == 2:
    h,l = 1<<int(np.ceil(np.log2(MAX_THREADS_BLOCK)/2)), 1<<int(np.floor(np.log2(MAX_THREADS_BLOCK)/2))
    block = (l if shape[0]>l else shape[0], h if shape[1]>h else shape[1], 1)
    d1,r1 = divmod(shape[0], block[0])
    d2,r2 = divmod(shape[1], block[1])
    grid = (d1+(r1>0), d2+(r2>0))
  #TODO: make better algo for choosing block size for 3D tensors
  elif len(shape) == 3:
    c1,c2 = 1<<int(np.ceil(np.log2(MAX_THREADS_BLOCK)/3)), 1<<int(np.ceil(np.log2(MAX_THREADS_BLOCK)/3))
    block = (c1, c2, 3)
    d1,r1 = divmod(shape[1], block[0])
    d2,r2 = divmod(shape[2], block[1])
    grid = (d1+(r1>0), d2+(r2>0), 3)
  else: assert False, f"Invalid tensor shape: {shape}"
  return block, grid

def unary_op(code, x):
  if len(x.shape) == 1:
    M,N = 1,x.shape[0]
  elif len(x.shape) == 2:
    M,N = x.shape
  elif len(x.shape) == 3:
    M,N = x.shape[1],x.shape[2]
  else: assert False, f"Invalid tensor shape: {x.shape}"
  M,N = np.int32(M),np.int32(N)

  block,grid = get_block_grid(x.shape)
  ret = buffer_new(x.shape)

  mod = SourceModule(f"""
  __global__ void unop(float *dest, float *a_g, int M, int N)
  {{
    const int i = blockDim.x*blockIdx.x+threadIdx.x;
    const int j = blockDim.y*blockIdx.y+threadIdx.y;
    const int k = threadIdx.z;
    //printf("k: %d, \\n", k);
    float a = a_g[i+M*j+M*N*k];
    //printf("i: %d, j: %d, k: %d, a: %f \\n", i,j,k,a);
    dest[i+M*j+M*N*k] = {code};
  }}
  """)
  unop = mod.get_function("unop")
  unop(ret.buf, x.buf, M, N, block=block, grid=grid)
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
        idx_exprs[j] = "idx_ret%d + d%d*(%s)" % (i, i, idx_exprs[j])

  return SourceModule(f"""
  __kernel void binop(__global const float *x_g, __global const float *y_g, __global float *res_g{args}")
  {{
    int gid0 = get_global_id(0);{compute_idx_rets}
    float a = x_g[{idx_exprs[0]}];
    float b = y_g[{idx_exprs[1]}];
    res_g[gid0] = {code};\n
  }}
  """)

def binary_op(code, x, y):
  shape_ret, dimlist, complist = binary_broadcast(x.shape, y.shape)
  prod_list = np.array(dimlist, dtype=i32)[-1::-1].cumprod(dtype=i32)[-1::-1] # take cumprod from back to front

  prg = get_binop_prg(code, tuple(complist))
  ret = buffer_new(ctx, shape_ret, zero=True)
  prg.binop([prod_list[0]] if len(dimlist) > 0 else [1], None, x.cl, y.cl, ret.cl, *dimlist, *(prod_list[1:]))
  return ret


def reduce_op(code, code2, x, axis=None, start="0.0"):
  print('start')
  axis =[1]
  if len(x.shape) == 1:
    M,N = 1,x.shape[0]
  elif len(x.shape) == 2:
    M,N = x.shape
  elif len(x.shape) == 3:
    M,N = x.shape[1],x.shape[2]
  else: assert False, f"Invalid tensor shape: {x.shape}"
  M,N = np.int32(M),np.int32(N)

  if axis is None:
    # full reduce
    osize = (1,1)
  else:
    osize = np.array(x.shape)
    osize[list(axis)] = 1
    if len(osize)>2:
        osize = list(filter(lambda x: x != 1, osize))
  ret = buffer_new(osize, True)
  if axis is None:
    ret.shape = (1,)

  block,grid = get_block_grid(x.shape)

  mod = SourceModule(f"""
  __global__ void unop(float *dest, float *a_g, int M, int N)
  {{
    float tmp[10];

    const int d_idx = threadIdx.x;
    //const int i = blockDim.x*blockIdx.x+threadIdx.x;
    //const int j = blockDim.y*blockIdx.y+threadIdx.y;
    //const int k = threadIdx.z;
    //const int d_idx = (i+M*j+M*N*k)/N;
    const int idx = (d_idx)*N;

    float t = 0;
    for (int i=idx; i < idx + N ; i++) {{ 
        float a = a_g[i];
        t += a;
    }}
    tmp[d_idx] = t;

    dest[d_idx] = tmp[d_idx];
  }}
  """)
  unop = mod.get_function("unop")
  unop(ret.buf, x.buf, M, N, block=(10,1,1), grid=(1,1))
  print(ret.toCPU())
  return ret

class ReLU(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return unary_op('max(a, (float)0.)', input)

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op('a * (b >= 0)', grad_output, input)

class Sum(Function):
  def forward(ctx, input, axis=None):
    ctx.save_for_backward(input)
    return reduce_op('out += a', 'out', input, axis=axis)

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op('a * (b >= 0)', grad_output, input)

if __name__ == "__main__":

  from tinygrad.tensor import Tensor, Device

  n = 10
  test = np.arange(n**2)-((n**2)//2)
  test = test.reshape(n,n)
  test = test.astype(np.float32)

  r1 = Tensor(test, device=Device.CUDA)
  r2 = r1.relu()
  print(r2.data.toCPU())
  r3 = r2.sum()
  print(r3.data.toCPU())
 # r2.backward()
 # print(r2)

  '''
def binary_op(code, x, y):
  if len(x.shape) == 1:
      M,N = 1,x.shape[0]
  elif len(x.shape) == 2:
      M,N = x.shape
  elif len(x.shape) == 3:
      M,N = x.shape[1],x.shape[2]
  else: assert False, f"Invalid tensor shape: {x.shape}"
  M,N = np.int32(M),np.int32(N)

  block,grid = block_grid(x.shape)
  ret = np.empty(x.shape).astype(np.float32)

  mod = SourceModule(f"""
  __global__ void unop(float *dest, float *a_g, float *a_g, int M, int N)
  {{
    const int i = blockDim.x*blockIdx.x+threadIdx.x;
    const int j = blockDim.y*blockIdx.y+threadIdx.y;
    const int k = threadIdx.z;
    //printf("k: %d, \\n", k);
    float a = a_g[i+M*j+M*N*k];
    float b = b_g[i+M*j+M*N*k];
    //printf("i: %d, j: %d, k: %d, a: %f \\n", i,j,k,a);
    dest[i+M*j+M*N*k] = {code};
  }}
  """)
  unop = mod.get_function("unop")
  unop(cuda.Out(ret), cuda.In(x), cuda.In(y), M, N, block=block, grid=grid)
  return ret


  from tinygrad.tensor import Tensor, Device
  n = 25
  test = np.arange(n**2)-((n**2)//2)
  test = test.reshape(n,n)
  tens = Tensor(test, device=Device.CUDA)
  r = ReLU()
  print(r.forward(test))



  n = 1000
  test = np.arange(3*n**2)
  test = test.reshape(3,n,n).astype(np.float32)-((n**2//2))
  test = test.astype(np.float32)

  k = unary_op('max(a, (float)0.)', test)
  print(test, test.shape)
  print("\n")
  print(k)
  '''
