#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#ifndef M_DIM
#define M_DIM 73728
#endif
#ifndef H_DIM
#define H_DIM 5888
#endif
#ifndef DY_STRIDE
#define DY_STRIDE 3072
#endif
#ifndef REAL_INTER
#define REAL_INTER 2880
#endif
#ifndef THREADS
#define THREADS 256
#endif
typedef short short2v __attribute__((ext_vector_type(2)));
constexpr int PAIRS = 16;
constexpr int BLOCKS_PER_ROW = H_DIM / 32;
constexpr float LIMIT = 7.0f;
constexpr float ALPHA = 1.702f;
constexpr float NEG_ALPHA_LOG2E = (float)(1.702 * -1.4426950408889634);
extern "C" __global__ __launch_bounds__(THREADS) void dswiglu_bwd_fp8_fast(
    __hip_bfloat16* __restrict__ dh, __hip_fp8_storage_t* __restrict__ dq, uint8_t* __restrict__ de,
    const __hip_bfloat16* __restrict__ h, const __hip_bfloat16* __restrict__ dy) {
  const long long block = (long long)blockIdx.x * THREADS + threadIdx.x;
  if (block >= (long long)M_DIM * BLOCKS_PER_ROW) return;
  const int row = block / BLOCKS_PER_ROW;
  const int hb = block - (long long)row * BLOCKS_PER_ROW;
  const int inter0 = hb * PAIRS;
  float dg[PAIRS], dl[PAIRS];
  float amax = 0.0f;
#pragma unroll
  for (int i=0; i<PAIRS; i++) {
    const int inter = inter0 + i;
    const uint32_t hp = reinterpret_cast<const uint32_t*>(h)[(long long)row * (H_DIM/2) + inter];
    const __hip_bfloat16 bg = __builtin_bit_cast(__hip_bfloat16, (uint16_t)hp);
    const __hip_bfloat16 bl = __builtin_bit_cast(__hip_bfloat16, (uint16_t)(hp >> 16));
    const float hg = (float)bg, hl = (float)bl;
    const float ga = inter < REAL_INTER ? (float)dy[(long long)row * DY_STRIDE + inter] : 0.0f;
    const float xg = fminf(hg, LIMIT), xl = fminf(fmaxf(hl, -LIMIT), LIMIT);
    const float sig = 1.0f / (1.0f + __builtin_amdgcn_exp2f(xg * NEG_ALPHA_LOG2E));
    const float sprime = sig * (1.0f + ALPHA * xg * (1.0f - sig));
    const float a = ga * sprime * (xl + 1.0f) * (hg < LIMIT ? 1.0f : 0.0f);
    const float b = ga * (xg * sig) * ((hl > -LIMIT && hl < LIMIT) ? 1.0f : 0.0f);
    const __hip_bfloat16 ab = (__hip_bfloat16)a, bb = (__hip_bfloat16)b;
    const float ar = (float)ab, br = (float)bb;
    dg[i] = ar; dl[i] = br;
    reinterpret_cast<uint32_t*>(dh)[(long long)row * (H_DIM/2) + inter] =
      (uint32_t)__builtin_bit_cast(uint16_t, ab) | ((uint32_t)__builtin_bit_cast(uint16_t, bb) << 16);
    amax = fmaxf(amax, fmaxf(fabsf(ar), fabsf(br)));
  }
  int e8 = (int)((__builtin_bit_cast(unsigned, amax) >> 23) & 0xffu);
  e8 = min(e8, 254);
  const float scale = __builtin_bit_cast(float, (unsigned)e8 << 23);
#pragma unroll
  for (int i=0; i<PAIRS; i++) {
    short2v p = {0, 0};
    p = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p, dg[i], dl[i], scale, false);
    const unsigned char *bytes = reinterpret_cast<const unsigned char*>(&p);
    reinterpret_cast<uint16_t*>(dq)[(long long)row * (H_DIM/2) + inter0 + i] =
      (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
  }
  de[block] = (uint8_t)e8;
}
