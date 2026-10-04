#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// One wave per row. Preserve the measured generated reduction's 45-element lane sums,
// followed by its lane order 0,16,32,48,1,17,...,15,31,47,63.
static __device__ __forceinline__ float saved_reciprocal(float x) {
  volatile float q = __builtin_amdgcn_rcpf(x);
  const unsigned bits = __builtin_bit_cast(unsigned, x);
  if ((bits & 0x7f800000u) == 0x7f800000u) return q;
  volatile float err = __builtin_fmaf(-x, q, 1.0f);
  return __builtin_fmaf(err, q, q);
}

extern "C" __global__ __launch_bounds__(256) void gptoss_rmsnorm_mul_fwd(
    __hip_bfloat16 *__restrict__ out, float *__restrict__ rrms,
    const __hip_bfloat16 *__restrict__ x, const __hip_bfloat16 *__restrict__ weight) {
  const int lane = threadIdx.x % 64, wave = threadIdx.x / 64;
  const int row = blockIdx.x * 4 + wave;
  __shared__ float partial[256];
  float values[45];
  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < 45; k++) {
    const float v = (float)x[(long long)row * 2880 + lane + k * 64];
    values[k] = v;
    acc = acc + v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  if (lane == 0) {
    float sum = 0.0f;
#pragma unroll
    for (int a = 0; a < 16; a++) {
#pragma unroll
      for (int b = 0; b < 4; b++) sum = sum + partial[wave * 64 + a + b * 16];
    }
    const float denom = __ocml_sqrt_f32(__builtin_fmaf(sum, 0.00034722223062999547f, EPS_LITERAL));
    const float r = saved_reciprocal(denom);
    rrms[row] = r;
    partial[wave * 64] = r;
  }
  __syncthreads();
  const float r = partial[wave * 64];
#pragma unroll
  for (int k = 0; k < 45; k++) {
    const int col = lane + k * 64;
    const float y = (values[k] * r) * (float)weight[col];
    out[(long long)row * 2880 + col] = __float2bfloat16(y);
  }
}
