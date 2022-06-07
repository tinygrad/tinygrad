import numpy as np

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

