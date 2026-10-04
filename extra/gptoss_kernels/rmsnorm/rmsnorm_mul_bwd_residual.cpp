#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#if !defined(ROWS) || !defined(HIDDEN) || !defined(PAD_D) || !defined(NUM_WG) || !defined(THREADS)
#error "ROWS, HIDDEN, PAD_D, NUM_WG and THREADS are required"
#endif

static_assert(THREADS % 64 == 0, "THREADS must contain whole AMD waves");
constexpr int COLS_PER_THREAD = (HIDDEN + THREADS - 1) / THREADS;

// GPT-OSS feed-forward RMSNorm backward. Reconstruct the exact BF16 residual input in registers from
// an explicitly physical [ROWS,PAD_D] projection. The two casts preserve the original ADD boundaries.
extern "C" __global__ __launch_bounds__(THREADS) void rmsnorm_mul_bwd_residual(
    __hip_bfloat16 *__restrict__ grad_x,
    float *__restrict__ grad_weight_partial,
    const __hip_bfloat16 *__restrict__ grad,
    const __hip_bfloat16 *__restrict__ residual,
    const __hip_bfloat16 *__restrict__ proj_phys,
    const __hip_bfloat16 *__restrict__ bias,
    const __hip_bfloat16 *__restrict__ weight,
    const float *__restrict__ rrms) {
  const int tid = threadIdx.x, part = blockIdx.x;
  float dw[COLS_PER_THREAD];
#pragma unroll
  for (int j = 0; j < COLS_PER_THREAD; j++) dw[j] = 0.0f;

  for (int row = part; row < ROWS; row += NUM_WG) {
    const float r = rrms[row];
    float xn[COLS_PER_THREAD], dy[COLS_PER_THREAD], dxn[COLS_PER_THREAD];
    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; j++) {
      const int col = tid + j * THREADS;
      if (col < HIDDEN) {
        const long long idx = (long long)row * HIDDEN + col;
        const long long pidx = (long long)row * PAD_D + col;
        const __hip_bfloat16 attn = (__hip_bfloat16)((float)proj_phys[pidx] + (float)bias[col]);
        const __hip_bfloat16 h = (__hip_bfloat16)((float)residual[idx] + (float)attn);
        xn[j] = (float)h * r;
        dy[j] = (float)grad[idx];
        dxn[j] = dy[j] * (float)weight[col];
        dot = fmaf(dxn[j], xn[j], dot);
      }
    }

#pragma unroll
    for (int off = 32; off; off >>= 1) dot += __shfl_down(dot, off, 64);
    __shared__ float wave_sums[THREADS / 64];
    const int lane = tid & 63, wave = tid >> 6;
    if (lane == 0) wave_sums[wave] = dot;
    __syncthreads();
    if (wave == 0) {
      dot = lane < THREADS / 64 ? wave_sums[lane] : 0.0f;
#pragma unroll
      for (int off = 32; off; off >>= 1) dot += __shfl_down(dot, off, 64);
      if (lane == 0) wave_sums[0] = dot;
    }
    __syncthreads();
    const float mean_dot = wave_sums[0] * (1.0f / (float)HIDDEN);

#pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; j++) {
      const int col = tid + j * THREADS;
      if (col < HIDDEN) {
        const long long idx = (long long)row * HIDDEN + col;
        grad_x[idx] = (__hip_bfloat16)(r * (dxn[j] - xn[j] * mean_dot));
        dw[j] = fmaf(dy[j], xn[j], dw[j]);
      }
    }
  }

#pragma unroll
  for (int j = 0; j < COLS_PER_THREAD; j++) {
    const int col = tid + j * THREADS;
    if (col < HIDDEN) grad_weight_partial[(long long)part * HIDDEN + col] = dw[j];
  }
}
