import os
import numpy as np
os.environ["CUDA"] = "1"
from tinygrad.runtime.ops_cuda import CUDAAllocator, CUDADevice, CUDAProgram, compile_cuda
from tinygrad.helpers import flat_mv

FLOAT16 = True
ACC_FLOAT16 = False
N = 4096

na = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32)
nb = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32)
nc = np.empty(N*N, np.float32)

# if FLOAT16:
#   na = na.astype(np.float16)
#   nb = nb.astype(np.float16)

device = CUDADevice("cuda:0")
cudaalloc = CUDAAllocator(device)

a = cudaalloc.alloc(N*N*4)
b = cudaalloc.alloc(N*N*4)
c = cudaalloc.alloc(N*N*4)

cudaalloc.copyin(a, bytearray(na))
cudaalloc.copyin(b, bytearray(nb))

FLOPS = N*N*N*2
BW = N*N*3*4

prog = CUDAProgram(device, "wmma_example", compile_cuda(f"""
#include <mma.h>
using namespace nvcuda;

#define MATRIX_M 16384
#define MATRIX_N 16384
#define MATRIX_K 16384

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


extern "C" __global__ void wmma_example(half *a, half *b, float *c,
                                        int M, int N, int K,
                                        float alpha, float beta)
{{
    int lda = M;
    int ldb = K;
    int ldc = M;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int i = 0; i < K; i += WMMA_K) {{
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {{
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }}
    }}

   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {{
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

#pragma unroll
      for(int i=0; i < c_frag.num_elements; i++) {{
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }}


      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }}
}}
"""))

global_size, local_size = [(N//16)//4, (N//16)//4, 1], [32, 1, 1]
tm = min([prog(a, b, c, global_size=global_size, local_size=local_size, wait=True) for _ in range(20)])
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul, {BW*1e-9/tm:.2f} GB/s")
cudaalloc.copyout(flat_mv(nc.data), c)
np.testing.assert_allclose(na.T.astype(np.float32) @ nb.T.astype(np.float32), nc.reshape(N,N).T, atol=1e-2)