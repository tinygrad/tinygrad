#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#if !defined(T_DIM) || !defined(M_DIM) || !defined(D_DIM) || !defined(P_DIM) || !defined(K_DIM) || !defined(THREADS)
#error "T_DIM, M_DIM, D_DIM, P_DIM, K_DIM and THREADS are required"
#endif
static_assert(K_DIM == 4 && D_DIM == 2880 && P_DIM == 3072);

// One workgroup owns a token. Load its four routing rows/weights once, then cooperatively stream
// the four gathered BF16 rows and packed BF16 output. This replaces 45 workgroups per token.
extern "C" __global__ __launch_bounds__(THREADS) void combine_fwd_token(
    __hip_bfloat16 *__restrict__ out,
    const __hip_bfloat16 *__restrict__ z,
    const int *__restrict__ dest_row,
    const float *__restrict__ weights) {
  const int t = blockIdx.x, tid = threadIdx.x;
  __shared__ int rows[K_DIM];
  __shared__ float ws[K_DIM];
  if (tid < K_DIM) {
    rows[tid] = dest_row[t * K_DIM + tid];
    ws[tid] = weights[t * K_DIM + tid];
  }
  __syncthreads();

  for (int d = tid; d < D_DIM; d += THREADS) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < K_DIM; j++) acc = fmaf(ws[j], (float)z[(long long)rows[j] * P_DIM + d], acc);
    out[(long long)t * D_DIM + d] = (__hip_bfloat16)acc;
  }
}
