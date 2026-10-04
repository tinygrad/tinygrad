#include <hip/hip_runtime.h>
#include <math.h>

#ifndef GEMM_M
#define GEMM_M 16384
#endif
#ifndef GEMM_N
#define GEMM_N 128256
#endif
#ifndef FINALIZE_THREADS
#define FINALIZE_THREADS 16
#endif

constexpr int TILES_N = GEMM_N / 256;
static_assert(GEMM_M == 16384 && GEMM_N == 128256 && TILES_N == 501);
static_assert(GEMM_M % FINALIZE_THREADS == 0);

// Fuse the two rowwise reductions used by the BF16 GPT-OSS LM-head. Each thread retains the exact scalar
// reduction order emitted by tinygrad's JITBEAM=3 kernels; putting both passes in one launch lets the second
// pass reuse the just-read partial maxima from cache and removes one dispatch without changing the tensor ABI.
extern "C" __global__ __launch_bounds__(FINALIZE_THREADS) void lmhead_ce_finalize_bf16(
    float *lse, float *partial_max, float *partial_sum) {
  const int row = blockIdx.x * FINALIZE_THREADS + threadIdx.x;
  const int base = row * TILES_N;

  float max_buf[1];
  *(max_buf + 0) = __builtin_bit_cast(float, 0xff800000u);
  for (int j = 0; j < TILES_N; j++) {
    const float x = partial_max[base + j];
    const float accum = *(max_buf + 0);
    // Match the generated `(acc < x) ? x : acc`, including its NaN and signed-zero behavior.
    *(max_buf + 0) = accum < x ? x : accum;
  }

  const float row_max = *(max_buf + 0);
  float sum_buf[1];
  *(sum_buf + 0) = 0.0f;
  for (int j = 0; j < TILES_N; j++) {
    const float sum = *(sum_buf + 0);
    *(sum_buf + 0) = sum + partial_sum[base + j] *
      exp2f((partial_max[base + j] - row_max) * 1.4426950216293335f);
  }
  lse[row] = row_max + log2f(*(sum_buf + 0)) * 0.6931471824645996f;
}
