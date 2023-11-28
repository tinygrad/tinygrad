import os
os.environ["METAL"] = "1"
import time
import numpy as np
from tinygrad.helpers import dtypes, getenv
from tinygrad.runtime.ops_metal import RawMetalBuffer, MetalProgram, compile_metal

N = getenv("N", 2048)
LID = 2

a = RawMetalBuffer(N*N, dtypes.float32)

nb = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32) #.astype(np.int32).astype(np.float32)
nc = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32) #.astype(np.int32).astype(np.float32)
b = RawMetalBuffer.fromCPU(nb)
c = RawMetalBuffer.fromCPU(nc)

FLOPS = N*N*N*2
BW = N*N*3*4

prog = MetalProgram("test", compile_metal(f"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+
using namespace metal;
kernel void test(device float *a, device const float *data1, device const float *data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
  a += gid.x * 32 * {N} + (gid.y * {LID} + lid.y) * 32;
  data1 += gid.x * 32 * {N};
  data2 += (gid.y * {LID} + lid.y) * 32;

  simdgroup_float8x8 acc[4][4];
  #pragma unroll(4)
  for (uint i = 0; i < 4; i++) {{
    #pragma unroll(4)
    for (uint j = 0; j < 4; j++) {{
      acc[i][j] = simdgroup_float8x8(0);
    }}
  }}

  simdgroup_float8x8 A[4];
  simdgroup_float8x8 B[4];
  for (uint k = 0; k < {N}; k+=8) {{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll(4)
    for (int i = 0; i < 4; ++i) {{
      simdgroup_load(A[i], data1+k+i*8*{N}, {N}, ulong2(0, 0));
      simdgroup_load(B[i], data2+i*8+k*{N}, {N}, ulong2(0, 0));
    }}

    #pragma unroll(4)
    for (int i = 0; i < 4; ++i) {{
      #pragma unroll(4)
      for (int j = 0; j < 4; ++j) {{
          simdgroup_multiply_accumulate(acc[i][j], A[j], B[i], acc[i][j]);
      }}
    }}
  }}
  #pragma unroll(4)
  for (int i = 0; i < 4; ++i) {{
    #pragma unroll(4)
    for (int j = 0; j < 4; ++j) {{
      simdgroup_store(acc[j][i], a+j*8+i*8*{N}, {N}, ulong2(0, 0));
    }}
  }}
}}"""))
def timeit(fxn):
  st = time.perf_counter()
  et = fxn()
  # NOTE: et doesn't contain the launch overhead
  return time.perf_counter() - st
tm = min([timeit(lambda: prog(a, b, c, global_size=[N//(8*4), N//(8*4*LID), 1], local_size=[32, LID, 1], wait=True)) for _ in range(20)])
na = a.toCPU().reshape(N,N)
comp = nb@nc
if N <= 32:
  print(na)
  print(comp)
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul, {BW*1e-9/tm:.2f} GB/s")
np.testing.assert_allclose(na, comp, atol=1e-3)

import torch, torch.mps
b = torch.from_numpy(nb).to('mps')
c = torch.from_numpy(nc).to('mps')

def torch_prog(b, c):
  st = time.perf_counter()
  a = b@c
  torch.mps.synchronize()
  return time.perf_counter() - st
tm = min([torch_prog(b, c) for _ in range(20)])
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul in torch")

from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.runtime.ops_metal import METAL
b = Tensor(nb)
c = Tensor(nc)
# TODO: slowness without the JIT I suspect comes from a lack of a caching allocator
@TinyJit
def tiny_jit(b, c):
  return (b@c).realize()
def tiny_prog(b, c):
  st = time.perf_counter()
  a = tiny_jit(b, c)
  METAL.synchronize()
  return time.perf_counter() - st
tm = min([tiny_prog(b, c) for _ in range(20)])
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul in tinygrad")
