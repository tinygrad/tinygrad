import os, time
import numpy as np
os.environ["CUDA"] = "1"
from tinygrad.runtime.ops_cuda import CUDAAllocator, CUDADevice, CUDAProgram, CUDACompiler
from tinygrad.helpers import flat_mv

dims = (36, 120087, 268)
N = 1
for d in dims:
  N = N*d

nx = np.zeros(dims).astype(np.float32)
nout = np.zeros(dims).astype(np.float32)

print('ARG_DTYPES:', nx.dtype)

device = CUDADevice("cuda:0")
cudaalloc = CUDAAllocator(device)

x = cudaalloc.alloc(N*4)

cudaalloc.copyin(x, bytearray(nx))


print(device.arch)
compiler = CUDACompiler(device.arch)

prog = CUDAProgram(device, "transfer_example", compiler.compile(f"""

extern "C" __global__ void transfer_example(float *x){{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  x[i] = 1.23f+x[i];
}}

"""))

global_size, local_size = [N//1000,1,1], [1000, 1, 1]
times, runs = 0, 10
for i in range(runs):
  prog(x, global_size=global_size, local_size=local_size, wait=True)
  st = time.perf_counter()
  cudaalloc.copyout(flat_mv(nout.data), x)
  et = time.perf_counter()
  times+=(et-st)

print(f'Ran {runs} times || Kernel AVG Runtime {times/runs*1000} ms')
print(nx[0])
print(nout[0])
