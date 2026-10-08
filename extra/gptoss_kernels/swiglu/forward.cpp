#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef unsigned uint4v __attribute__((ext_vector_type(4)));
typedef unsigned short ushort4v __attribute__((ext_vector_type(4)));
typedef short short2v __attribute__((ext_vector_type(2)));

__device__ inline float bf16(float x) { return (float)(__hip_bfloat16)x; }

extern "C" __global__ __launch_bounds__(256) void gptoss_swiglu_quantize(
    unsigned char *__restrict__ q, unsigned char *__restrict__ scales, const unsigned *__restrict__ h) {
  // Four lanes per MX block, each loading eight interleaved gate/linear pairs.
  const int block = blockIdx.x*64 + threadIdx.x/4, lane = threadIdx.x%4;
  const int row = block/96, col = (block%96)*32 + lane*8;
  if (col >= 2880) {
    *reinterpret_cast<ushort4v*>(&q[block*32 + lane*8]) = ushort4v{};
    if (lane == 0) scales[block] = 0;
    return;
  }
  const uint4v lo = *reinterpret_cast<const uint4v*>(&h[row*2880 + col]);
  const uint4v hi = *reinterpret_cast<const uint4v*>(&h[row*2880 + col + 4]);
  float values[8], amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const unsigned bits = i < 4 ? lo[i] : hi[i-4];
    const float gate = __builtin_bit_cast(float, bits << 16), up = __builtin_bit_cast(float, bits & 0xffff0000u);
    const float glu = gate > 7.0f ? 7.0f : gate;
    const float linear = up > 7.0f ? 7.0f : (up < -7.0f ? -7.0f : up);
    // Stock codegen folds alpha * -log2(e) in BF16. Keep its remaining rounding boundaries.
    const float sig = bf16(1.0f / bf16(1.0f + bf16(exp2f(bf16(glu * -2.453125f)))));
    values[i] = bf16(bf16(glu * sig) * bf16(linear + 1.0f));
    amax = fmaxf(amax, fabsf(values[i]));
  }
#pragma unroll
  for (int offset = 2; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_down(amax, offset, 4));
  amax = __shfl(amax, 0, 4);
  const unsigned e8 = min((__builtin_bit_cast(unsigned, amax) >> 23) & 255u, 254u);
  if (lane == 0) scales[block] = e8;
  ushort4v packed;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    short2v pair = {0, 0};
    pair = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pair, values[i*2], values[i*2+1], __builtin_bit_cast(float, e8 << 23), false);
    packed[i] = __builtin_bit_cast(unsigned short, pair[0]);
  }
  *reinterpret_cast<ushort4v*>(&q[block*32 + lane*8]) = packed;
}
