import os, time
import numpy as np
os.environ["CUDA"] = "1"
from tinygrad.runtime.ops_cuda import CUDAAllocator, CUDADevice, CUDAProgram, CUDACompiler
from tinygrad.helpers import flat_mv

N = 10000000
TRUE_CNT = 2000
print(f'TRUES: {TRUE_CNT} || FALSES: {N-TRUE_CNT}')
nx = np.zeros((N))
ny = np.zeros((N))-1
for i in range(TRUE_CNT):
  nx[i] = 1
# np.random.shuffle(nx)

nx = nx.astype(bool)
ny = ny.astype(np.int32)
print('ARG_DTYPES:', nx.dtype, ny.dtype)

device = CUDADevice("cuda:0")
cudaalloc = CUDAAllocator(device)

x = cudaalloc.alloc(N*4)
y = cudaalloc.alloc(N*4)

cudaalloc.copyin(x, bytearray(nx))
cudaalloc.copyin(y, bytearray(ny))

print(device.arch)
compiler = CUDACompiler(device.arch)

prog = CUDAProgram(device, "boolidx_example", compiler.compile(f"""

__device__ unsigned int count=0;

extern "C" __global__ void boolidx_example(bool *x, int *y){{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(x[i]==1){{
    int y_ind = atomicAdd(&count, 1);
    y[y_ind] = i;
  }}
}}

"""))

global_size, local_size = [10000,1,1], [1000, 1, 1]
st = time.perf_counter()
prog(x, y, global_size=global_size, local_size=local_size, wait=True)
et = time.perf_counter()

print(f'Kernel runtime {(et-st)*1000} ms')
cudaalloc.copyout(flat_mv(ny.data), y)
np.testing.assert_equal(np.sort(ny[:TRUE_CNT]), np.arange(TRUE_CNT))
print(f'NUM OF TRUES MATCH: {(ny>=0).sum()==TRUE_CNT}')