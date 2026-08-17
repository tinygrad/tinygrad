#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#if WRITE_MXFP4_ROW
#include "quantize_mxfp4_device.h"
#endif

#ifndef WRITE_MXFP4_ROW
#define WRITE_MXFP4_ROW 0
#endif

constexpr int VEC = 8;
static_assert(HIDDEN % (THREADS * VEC) == 0);
constexpr int VECS_PER_THREAD = HIDDEN / (THREADS * VEC);

extern "C" __global__ __launch_bounds__(THREADS) void rmsnorm_add_mul_fwd(
    __hip_bfloat16* __restrict__ out, __hip_bfloat16* __restrict__ h_out, float* __restrict__ rrms,
    const __hip_bfloat16* __restrict__ x, const __hip_bfloat16* __restrict__ residual,
    const __hip_bfloat16* __restrict__ weight
#if WRITE_MXFP4_ROW
    , uint8_t* __restrict__ row_fp4, uint8_t* __restrict__ row_scale
#endif
    ) {
  __shared__ float sums[THREADS];
#if WRITE_MXFP4_ROW
  __shared__ uint16_t quant_row[HIDDEN];
#endif
  const int tid = threadIdx.x;
  float w[VECS_PER_THREAD * VEC];
  #pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; v++) {
    const int h = tid * VEC + v * THREADS * VEC;
    float4 raw = *reinterpret_cast<const float4*>(&weight[h]);
    const __hip_bfloat16 *p = reinterpret_cast<const __hip_bfloat16*>(&raw);
    #pragma unroll
    for (int i = 0; i < VEC; i++) w[v*VEC+i] = static_cast<float>(p[i]);
  }
  for (int row = blockIdx.x; row < ROWS; row += NUM_WG) {
    float hv[VECS_PER_THREAD * VEC], sum = 0.0f;
    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
      const int h = tid * VEC + v * THREADS * VEC;
      float4 xr = *reinterpret_cast<const float4*>(&x[row*HIDDEN+h]);
      float4 rr = *reinterpret_cast<const float4*>(&residual[row*HIDDEN+h]);
      const __hip_bfloat16 *xp = reinterpret_cast<const __hip_bfloat16*>(&xr);
      const __hip_bfloat16 *rp = reinterpret_cast<const __hip_bfloat16*>(&rr);
      __hip_bfloat16 hb[VEC];
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        hb[i] = __hip_bfloat16(static_cast<float>(xp[i]) + static_cast<float>(rp[i]));
        float q = static_cast<float>(hb[i]); hv[v*VEC+i] = q; sum += q*q;
      }
      *reinterpret_cast<float4*>(&h_out[row*HIDDEN+h]) = *reinterpret_cast<float4*>(hb);
    }
    sums[tid] = sum;
    __syncthreads();
    for (int s = THREADS/2; s; s >>= 1) { if (tid < s) sums[tid] += sums[tid+s]; __syncthreads(); }
    const float rs = rsqrtf(sums[0] / float(HIDDEN) + EPS);
    if (tid == 0) rrms[row] = rs;
    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
      const int h = tid * VEC + v * THREADS * VEC;
      __hip_bfloat16 y[VEC];
      #pragma unroll
      for (int i = 0; i < VEC; i++) y[i] = __hip_bfloat16(hv[v*VEC+i] * rs * w[v*VEC+i]);
      *reinterpret_cast<float4*>(&out[row*HIDDEN+h]) = *reinterpret_cast<float4*>(y);
#if WRITE_MXFP4_ROW
      *reinterpret_cast<float4*>(&quant_row[h]) = *reinterpret_cast<float4*>(y);
#endif
    }
    __syncthreads();
#if WRITE_MXFP4_ROW
    #pragma unroll
    for (int q = 0; q < HIDDEN / (THREADS * 4); q++) {
      const int col = q * THREADS * 4 + tid * 4;
      const int lane = tid & 7;
      const auto result = mxfp4::quantize(mxfp4::load_bf16x4(quant_row + col), lane);
      mxfp4::store_fp4<false>(row_fp4, row, col / 2, HIDDEN / 2, result.fp4);
      if (lane == 0) mxfp4::store_scale(row_scale, row, col / 32, HIDDEN / 32, result.scale);
    }
    __syncthreads();
#endif
  }
}
