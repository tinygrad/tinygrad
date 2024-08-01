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
np.random.shuffle(nx)

nx = nx.astype(bool)
ny = ny.astype(np.int32)
zero = np.zeros((1)).astype(np.uint32)
print('ARG_DTYPES:', nx.dtype, ny.dtype, zero.dtype)
idx_real = np.sort(np.nonzero(nx)[0])

device = CUDADevice("cuda:0")
cudaalloc = CUDAAllocator(device)

x = cudaalloc.alloc(N*4)
y = cudaalloc.alloc(N*4)
cnt_ptr = cudaalloc.alloc(4)

cudaalloc.copyin(x, bytearray(nx))
cudaalloc.copyin(y, bytearray(ny))
cudaalloc.copyin(cnt_ptr, bytearray(zero))

print(device.arch)
compiler = CUDACompiler(device.arch)

prog = CUDAProgram(device, "boolidx_example", compiler.compile(f"""

extern "C" __global__ void boolidx_example(bool *x, int *y, unsigned int *cnt_ptr){{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(x[i]==1){{
    int y_ind = atomicAdd(cnt_ptr, 1);
    y[y_ind] = i;
  }}
}}

"""))

global_size, local_size = [10000,1,1], [1000, 1, 1]
times, runs = 0, 100
for i in range(runs):
  st = time.perf_counter()
  prog(x, y, cnt_ptr, global_size=global_size, local_size=local_size, wait=True)
  cudaalloc.copyin(cnt_ptr, bytearray(zero))
  et = time.perf_counter()
  times+=(et-st)

print(f'Ran {runs} times || Kernel AVG Runtime {times/runs*1000} ms')
cudaalloc.copyout(flat_mv(ny.data), y)
np.testing.assert_equal(np.sort(ny[:TRUE_CNT]), idx_real)
print(f'NUM OF TRUES MATCH: {(ny>=0).sum()==TRUE_CNT}')
print(f'Rest of array == -1: {ny[TRUE_CNT:].mean()==-1}')