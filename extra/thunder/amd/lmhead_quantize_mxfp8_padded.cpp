#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>
#ifndef ROWS
#define ROWS 16384
#endif
#ifndef H_DIM
#define H_DIM 2880
#endif
#ifndef K_DIM
#define K_DIM 3072
#endif
constexpr int THREADS=256, BLK=32, PACK=4, SUPER=BLK*PACK, GROUPS=K_DIM/SUPER;
static_assert(H_DIM<=K_DIM && H_DIM%BLK==0 && K_DIM%SUPER==0);
typedef short short2v __attribute__((ext_vector_type(2)));

extern "C" __global__ __launch_bounds__(THREADS) void lmhead_quantize_mxfp8_padded(
    __hip_fp8_storage_t *__restrict__ q, uint32_t *__restrict__ si,
    const __hip_bfloat16 *__restrict__ x) {
  const int super=blockIdx.x*THREADS+threadIdx.x;
  if(super>=ROWS*GROUPS) return;
  const int row=super/GROUPS, group=super-row*GROUPS, kbase=group*SUPER;
  uint32_t packed=0;
  #pragma unroll
  for(int sb=0;sb<PACK;sb++) {
    const int k0=kbase+sb*BLK;
    __hip_fp8_storage_t out[BLK];
    int e8=0;
    if(k0<H_DIM) {
      float vals[BLK], amax=0.0f;
      #pragma unroll
      for(int j=0;j<BLK;j++) { const float v=(float)x[(long long)row*H_DIM+k0+j]; vals[j]=v; amax=fmaxf(amax,fabsf(v)); }
      e8=min(254,(int)((__builtin_bit_cast(unsigned,amax)>>23)&0xffu));
      if(amax==0.0f) {
        #pragma unroll
        for(int j=0;j<BLK;j++) out[j]=0;
      } else {
        const float sf=__builtin_bit_cast(float,(unsigned)e8<<23);
        #pragma unroll
        for(int j=0;j<BLK;j+=2) {
          short2v acc={0,0}; acc=__builtin_amdgcn_cvt_scalef32_pk_fp8_f32(acc,vals[j],vals[j+1],sf,false);
          const unsigned char *rb=reinterpret_cast<const unsigned char*>(&acc); out[j]=rb[0]; out[j+1]=rb[1];
        }
      }
    } else {
      #pragma unroll
      for(int j=0;j<BLK;j++) out[j]=0;
    }
    const long long obase=(long long)row*K_DIM+k0;
    *reinterpret_cast<uint4*>(&q[obase])=*reinterpret_cast<uint4*>(&out[0]);
    *reinterpret_cast<uint4*>(&q[obase+16])=*reinterpret_cast<uint4*>(&out[16]);
    packed|=(uint32_t)(uint8_t)e8<<(8*sb);
  }
  si[(long long)group*ROWS+row]=packed;
}
