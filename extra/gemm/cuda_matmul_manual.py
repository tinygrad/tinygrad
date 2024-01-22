import os
import numpy as np
from tinygrad.helpers import flat_mv
from tinygrad.runtime.ops_cuda import CUDAAllocator, CUDADevice, CUDAProgram, compile_cuda

np.set_printoptions(precision=1, suppress=True, threshold=1000000, linewidth=9999999)

N = int(os.environ.get("N", 4096))

na = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32).astype(np.float16)
nb = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32).astype(np.float16)

FLOPS = N*N*N*2
BW = N*N*3*4

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16

local_size = [16,2,1]
warp_size = [1,1,1]

warp_dims_left = 32
for i in range(len(warp_size)):
  w = min(local_size[i], warp_dims_left)
  warp_size[i] = int(w)
  warp_dims_left /= w
  if warp_dims_left <= 1:
    break

gsize0 = N / (WMMA_M * local_size[0] / warp_size[0])
gsize1 = N / (WMMA_N * local_size[1] / warp_size[1])

gsize0 = max(1, int(np.ceil(gsize0)))
gsize1 = max(1, int(np.ceil(gsize1)))

gsize2 = 1

def check_gsize(g):
  global gsize2
  if g > 1024:
    gsize2 *= int(np.ceil(g/1024))
    g = 1024
  return g

global_size = [check_gsize(gsize0), check_gsize(gsize1), gsize2]

nc = np.zeros(N*N, np.float32)

device = CUDADevice("cuda:0")
cudaalloc = CUDAAllocator(device)

a = cudaalloc.alloc(N*N*2)
b = cudaalloc.alloc(N*N*2)
c = cudaalloc.alloc(N*N*4)

cudaalloc.copyin(a, bytearray(na.astype(np.float16)))
cudaalloc.copyin(b, bytearray(nb.astype(np.float16)))


code = f"""
#include <mma.h>
using namespace nvcuda;

struct __align__(16) float8 {{
  float4 val1;
  float4 val2;
}};

__device__ float8 make_float8(float a1, float a2, float a3, float a4, float b1, float b2, float b3, float b4) {{
  float8 result;
  result.val1 = make_float4(a1, a2, a3, a4);
  result.val2 = make_float4(b1, b2, b3, b4);
  return result;
}}

struct half4 {{
  half x, y, z, w;
}};

__device__ half4 make_half4(half x, half y, half z, half w) {{
  half4 result;
  result.x = x;
  result.y = y;
  result.z = z;
  result.w = w;
  return result;
}}

struct __align__(16) half8 {{
  half4 val1;
  half4 val2;
}};

__device__ half8 make_half8(half a1, half a2, half a3, half a4, half b1, half b2, half b3, half b4) {{
  half8 result;
  result.val1 = make_half4(a1, a2, a3, a4);
  result.val2 = make_half4(b1, b2, b3, b4);
  return result;
}}

extern "C" __global__ void simple_matmul(half *a, half *b, float *c) {{

  int t = (blockDim.z * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / {warp_size[0]};
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y) / {warp_size[1]};

  // Declare fragments
  half8 a_frag = make_half8(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
  half8 b_frag = make_half8(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
  float8 c_frag = make_float8(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);

  // Loop through tiles
  for (int i = 0; i < {N}; i += {WMMA_K}) {{
    int aTileRow = warpM * {WMMA_M};
    int aTileCol = i;
    int bTileRow = i;
    int bTileCol = warpN * {WMMA_N};

    // Load inputs (wmma::load_matrix_sync)
    for (int chunk = 0; chunk < 4; chunk++) {{
      int aRow = (t * 2 / 8) + (chunk % 2) * 8;
      int aCol = (t * 2 % 8) + (chunk / 2) * 8;
      int bRow = (t * 2 % 8) + (chunk % 2) * 8;
      int bCol = (t * 2 / 8) + (chunk / 2) * 8;
      ((half*)&a_frag)[chunk*2]   = *(a+(aTileRow+aRow)  *{N}+aTileCol+aCol);
      ((half*)&a_frag)[chunk*2+1] = *(a+(aTileRow+aRow)  *{N}+aTileCol+aCol+1);
      ((half*)&b_frag)[chunk*2]   = *(b+(bTileRow+bRow)  *{N}+bTileCol+bCol);
      ((half*)&b_frag)[chunk*2+1] = *(b+(bTileRow+bRow+1)*{N}+bTileCol+bCol);
    }}

    // Do the thing (wmma::mma_sync)
    __hmma_m16n16k16_mma_f32f32((float*)&c_frag, (const int*)&a_frag, (const int*)&b_frag, (const float*)&c_frag, 0, 0);
  }}

  int warpRow = warpM * {WMMA_M};
  int warpcol = warpN * {WMMA_N};

  // Store output (wmma::store_matrix_sync)
  for (int chunk = 0; chunk < 4; chunk++) {{
    int row = (t * 2 / 8) + (chunk % 2) * 8;
    int col = (t * 2 % 8) + (chunk / 2) * 8;
    *(c+(warpRow+row)*{N}+warpcol+col) = ((float*)&c_frag)[chunk*2];
    *(c+(warpRow+row)*{N}+warpcol+col+1) = ((float*)&c_frag)[chunk*2+1];
  }}
}}
"""

compiled = compile_cuda(code)
prog = CUDAProgram(device, "simple_matmul", compiled)

tm = min([prog(a, b, c, global_size=global_size, local_size=local_size, wait=True) for _ in range(1)])
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul, {BW*1e-9/tm:.2f} GB/s")
cudaalloc.copyout(flat_mv(nc.data), c)
nc = nc.reshape(N,N)
correct = na.astype(np.float32) @ nb.astype(np.float32)

np.testing.assert_allclose(correct, nc, atol=.01)