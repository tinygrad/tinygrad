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
constexpr int ELEMS_PER_THREAD = HIDDEN / THREADS;
constexpr int VECS_PER_THREAD = ELEMS_PER_THREAD / VEC;

extern "C" __global__ __launch_bounds__(THREADS) void rmsnorm_add_mul_bwd(
    __hip_bfloat16* __restrict__ dh, float* __restrict__ dweight_partial,
#if WRITE_MXFP4_ROW
    uint8_t* __restrict__ row_fp4, uint8_t* __restrict__ row_scale,
#endif
    const __hip_bfloat16* __restrict__ dout, const __hip_bfloat16* __restrict__ dh_direct,
    const __hip_bfloat16* __restrict__ h_in, const float* __restrict__ rrms,
    const __hip_bfloat16* __restrict__ weight) {
  __shared__ float sums[THREADS];
#if WRITE_MXFP4_ROW
  __shared__ uint16_t quant_row[HIDDEN];
#endif
  const int tid = threadIdx.x, wg = blockIdx.x;
  float w[ELEMS_PER_THREAD], dw[ELEMS_PER_THREAD] = {};
  #pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; v++) {
    const int h = tid * VEC + v * THREADS * VEC;
    float4 raw = *reinterpret_cast<const float4*>(&weight[h]);
    const __hip_bfloat16 *p = reinterpret_cast<const __hip_bfloat16*>(&raw);
    #pragma unroll
    for (int i = 0; i < VEC; i++) w[v*VEC+i] = static_cast<float>(p[i]);
  }
  for (int row = wg; row < ROWS; row += NUM_WG) {
    const float rs = rrms[row];
    float y[ELEMS_PER_THREAD], dy[ELEMS_PER_THREAD], direct[ELEMS_PER_THREAD], dot = 0.0f;
    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
      const int h = tid * VEC + v * THREADS * VEC;
      float4 xr = *reinterpret_cast<const float4*>(&h_in[row*HIDDEN+h]);
      float4 gr = *reinterpret_cast<const float4*>(&dout[row*HIDDEN+h]);
      float4 dr = *reinterpret_cast<const float4*>(&dh_direct[row*HIDDEN+h]);
      const __hip_bfloat16 *xp = reinterpret_cast<const __hip_bfloat16*>(&xr);
      const __hip_bfloat16 *gp = reinterpret_cast<const __hip_bfloat16*>(&gr);
      const __hip_bfloat16 *dp = reinterpret_cast<const __hip_bfloat16*>(&dr);
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        const int j=v*VEC+i; float dz=static_cast<float>(gp[i]);
        y[j]=static_cast<float>(xp[i])*rs; dy[j]=dz*w[j]; direct[j]=static_cast<float>(dp[i]);
        dw[j] += dz*y[j]; dot += dy[j]*y[j];
      }
    }
    sums[tid] = dot;
    __syncthreads();
    for (int s = THREADS/2; s; s >>= 1) { if (tid < s) sums[tid] += sums[tid+s]; __syncthreads(); }
    const float mean = sums[0] / float(HIDDEN);
    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
      const int h = tid * VEC + v * THREADS * VEC;
      __hip_bfloat16 o[VEC];
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        const int j = v*VEC+i;
        const __hip_bfloat16 rms_grad = __hip_bfloat16(rs*(dy[j]-y[j]*mean));
        o[i] = __hip_bfloat16(static_cast<float>(rms_grad) + direct[j]);
      }
      *reinterpret_cast<float4*>(&dh[row*HIDDEN+h]) = *reinterpret_cast<float4*>(o);
#if WRITE_MXFP4_ROW
      *reinterpret_cast<float4*>(&quant_row[h]) = *reinterpret_cast<float4*>(o);
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
  #pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; v++) {
    const int h = tid * VEC + v * THREADS * VEC;
    float4 lo{dw[v*VEC],dw[v*VEC+1],dw[v*VEC+2],dw[v*VEC+3]};
    float4 hi{dw[v*VEC+4],dw[v*VEC+5],dw[v*VEC+6],dw[v*VEC+7]};
    *reinterpret_cast<float4*>(&dweight_partial[wg*HIDDEN+h]) = lo;
    *reinterpret_cast<float4*>(&dweight_partial[wg*HIDDEN+h+4]) = hi;
  }
}
