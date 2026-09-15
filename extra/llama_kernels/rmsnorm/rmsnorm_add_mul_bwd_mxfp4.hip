#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "quantize_mxfp4_8.h"

// Preserve the original dot tree and strided weight gradient; quantize directly from registers.
static_assert(HIDDEN == 4096);
extern "C" __global__ __launch_bounds__(128)
void KERNEL_NAME(__hip_bfloat16* __restrict__ dh,float* __restrict__ dw,
                uint8_t* __restrict__ row_fp4,uint8_t* __restrict__ row_scale,
                const __hip_bfloat16* __restrict__ dout,const __hip_bfloat16* __restrict__ direct,
                const __hip_bfloat16* __restrict__ h,const float* __restrict__ rrms,const __hip_bfloat16* __restrict__ weight) {
  __shared__ float sums[128];
  const int tid=threadIdx.x,wg=blockIdx.x;
  float accum[32]={},w[32];
  #pragma unroll
  for (int v=0;v<4;v++) {
    const uint4 raw=*reinterpret_cast<const uint4*>(weight+tid*8+v*1024);
    const auto* p=reinterpret_cast<const __hip_bfloat16*>(&raw);
    #pragma unroll
    for (int i=0;i<8;i++) w[v*8+i]=float(p[i]);
  }
  #pragma unroll 1
  for (int row=wg;row<ROWS;row+=1024) {
    const float rs=rrms[row];
    float ys[32],dys[32],ds[32],dot=0;
    #pragma unroll
    for (int v=0;v<4;v++) {
      const int col=tid*8+v*1024,offset=row*HIDDEN+col;
      const uint4 hr=*reinterpret_cast<const uint4*>(h+offset),gr=*reinterpret_cast<const uint4*>(dout+offset),
                  dr=*reinterpret_cast<const uint4*>(direct+offset);
      const auto* hp=reinterpret_cast<const __hip_bfloat16*>(&hr);
      const auto* gp=reinterpret_cast<const __hip_bfloat16*>(&gr);
      const auto* dp=reinterpret_cast<const __hip_bfloat16*>(&dr);
      #pragma unroll
      for (int i=0;i<8;i++) {
        #pragma clang fp reassociate(off) contract(off)
        const int j=v*8+i;
        const float g=float(gp[i]);
        ys[j]=float(hp[i])*rs; dys[j]=g*w[j]; ds[j]=float(dp[i]);
        dot+=dys[j]*ys[j];
        accum[j]=__fmaf_rn(g,ys[j],accum[j]);
      }
    }
    sums[tid]=dot;
    __syncthreads();
    dot+=sums[tid^64];
    #pragma unroll
    for (int s=32;s;s>>=1) dot+=__shfl_xor(dot,s);
    const float mean=dot*(1.0f/HIDDEN);
    #pragma unroll
    for (int v=0;v<4;v++) {
      __hip_bfloat16 vals[8];
      #pragma unroll
      for (int i=0;i<8;i++) {
        const int j=v*8+i;
        const __hip_bfloat16 rms_grad=__hip_bfloat16(rs*__fmaf_rn(-ys[j],mean,dys[j]));
        vals[i]=__hip_bfloat16(float(rms_grad)+ds[j]);
      }
      const int col=tid*8+v*1024;
      *reinterpret_cast<uint4*>(dh+row*HIDDEN+col)=*reinterpret_cast<uint4*>(vals);
      const auto q=mxfp4::quantize8(mxfp4::load_bf16x4(reinterpret_cast<uint16_t*>(vals)),
        mxfp4::load_bf16x4(reinterpret_cast<uint16_t*>(vals)+4),tid%4);
      mxfp4::store_fp4_8<false>(row_fp4,row,col/2,HIDDEN/2,q.fp4);
      if (tid%4==0) mxfp4::store_scale(row_scale,row,col/32,HIDDEN/32,q.scale);
    }
    __syncthreads();
  }
  #pragma unroll
  for (int v=0;v<4;v++) {
    float* dest=dw+wg*HIDDEN+tid*8+v*1024;
    *reinterpret_cast<float4*>(dest)=make_float4(accum[v*8],accum[v*8+1],accum[v*8+2],accum[v*8+3]);
    *reinterpret_cast<float4*>(dest+4)=make_float4(accum[v*8+4],accum[v*8+5],accum[v*8+6],accum[v*8+7]);
  }
}
