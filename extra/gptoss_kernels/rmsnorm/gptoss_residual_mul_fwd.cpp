#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#ifndef GPTOSS_RESIDUAL_RMS_SHUFFLE
#define GPTOSS_RESIDUAL_RMS_SHUFFLE 0
#endif

// Two rows per wave, with adjacent BF16 pairs owned by each lane. Keep both 45-term
// accumulators and the original ordered lane sum: packing must not reassociate RMS math.
static __device__ __forceinline__ float saved_reciprocal(float x) {
  volatile float q = __builtin_amdgcn_rcpf(x);
  const unsigned bits = __builtin_bit_cast(unsigned, x);
  if ((bits & 0x7f800000u) == 0x7f800000u) return q;
  volatile float err = __builtin_fmaf(-x, q, 1.0f);
  return __builtin_fmaf(err, q, q);
}
static __device__ __forceinline__ float bf(unsigned bits) { return __builtin_bit_cast(float,bits << 16); }
extern "C" __global__ __launch_bounds__(64) void gptoss_residual_rmsnorm_mul_fwd(
    __hip_bfloat16 *__restrict__ out, float *__restrict__ rrms,
    __hip_bfloat16 *__restrict__ saved_h, const __hip_bfloat16 *__restrict__ x,
    const __hip_bfloat16 *__restrict__ proj, const __hip_bfloat16 *__restrict__ bias,
    const __hip_bfloat16 *__restrict__ weight) {
  const int lane = threadIdx.x % 32, row_local = threadIdx.x / 32;
  const int row = blockIdx.x * 2 + row_local;
#if !GPTOSS_RESIDUAL_RMS_SHUFFLE
  __shared__ float partial[128];
#endif
  float values[45][2];
  float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
  for (int k = 0; k < 45; k++) {
    const int col = lane + k * 32;
    const unsigned p = ((const unsigned*)proj)[row * 1536 + col];
    const unsigned b = ((const unsigned*)bias)[col];
    const unsigned xx = ((const unsigned*)x)[row * 1440 + col];
    const float v0 = (float)__float2bfloat16(bf(xx) + (float)__float2bfloat16(bf(p) + bf(b)));
    const float v1 = (float)__float2bfloat16(bf(xx >> 16) + (float)__float2bfloat16(bf(p >> 16) + bf(b >> 16)));
    values[k][0] = v0; values[k][1] = v1;
    acc0 = acc0 + v0*v0; acc1 = acc1 + v1*v1;
  }
#if GPTOSS_RESIDUAL_RMS_SHUFFLE
  // The original serial sum visits partial[a + 16*b], not a tree reduction.
  // Each half-wave independently broadcasts those exact same 64 addends in the same order.
  float sum = 0.0f;
#pragma unroll
  for (int a = 0; a < 16; a++) {
#pragma unroll
    for (int b = 0; b < 4; b++) sum = sum + __shfl((a & 1) ? acc1 : acc0, a/2 + b*8, 32);
  }
  const float denom = __ocml_sqrt_f32(__builtin_fmaf(sum, 0.00034722223062999547f, EPS_LITERAL));
  const float r = saved_reciprocal(denom);
  if (lane == 0) rrms[row] = r;
#else
  partial[threadIdx.x*2] = acc0; partial[threadIdx.x*2+1] = acc1;
  __syncthreads();
  if (lane == 0) {
    float sum = 0.0f;
#pragma unroll
    for (int a = 0; a < 16; a++) {
#pragma unroll
      for (int b = 0; b < 4; b++) sum = sum + partial[row_local * 64 + a + b * 16];
    }
    const float denom = __ocml_sqrt_f32(__builtin_fmaf(sum, 0.00034722223062999547f, EPS_LITERAL));
    const float r = saved_reciprocal(denom);
    rrms[row] = r;
    partial[row_local * 64] = r;
  }
  __syncthreads();
  const float r = partial[row_local * 64];
#endif
  // Delay both outputs until all residual input loads and the reduction have completed.
#pragma unroll
  for (int k = 0; k < 45; k++) {
    const int col = lane + k * 32;
    const unsigned w = ((const unsigned*)weight)[col];
    const float v0 = values[k][0], v1 = values[k][1];
    ((unsigned*)saved_h)[row*1440+col] = (__builtin_bit_cast(unsigned,v0)>>16) | (__builtin_bit_cast(unsigned,v1)&0xffff0000u);
    const unsigned y0 = __builtin_bit_cast(unsigned short,__float2bfloat16((v0*r)*bf(w)));
    const unsigned y1 = __builtin_bit_cast(unsigned short,__float2bfloat16((v1*r)*bf(w >> 16)));
    ((unsigned*)out)[row*1440+col] = y0 | (y1 << 16);
  }
}
