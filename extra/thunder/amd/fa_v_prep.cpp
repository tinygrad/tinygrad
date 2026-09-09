// Dynamic per-device V scaling: packed BF16 maxima, then fused final reduction and FP8 conversion.
#include <hip/hip_runtime.h>
typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
#if PARTIAL
__device__ __forceinline__ unsigned block_max(unsigned val) {
  __shared__ unsigned scratch[THREADS/64];
  #pragma unroll
  for (int offset=32; offset; offset/=2) val=max(val,__shfl_xor(val,offset,64));
  if ((threadIdx.x&63)==0) scratch[threadIdx.x/64]=val;
  __syncthreads();
  unsigned all=0;
  #pragma unroll
  for (int i=0;i<THREADS/64;i++) all=max(all,scratch[i]);
  return all;
}
extern "C" __global__ __launch_bounds__(THREADS) void fa_v_amax_partial(unsigned* out, const unsigned short* x) {
  u16x8 acc={};
  #pragma unroll
  for (int j=0;j<SIZE/(GROUPS*THREADS*8);j++) {
    int idx=(blockIdx.x*(SIZE/GROUPS)+(j*THREADS+threadIdx.x)*8);
    u16x8 val=*(const u16x8*)(x+idx);
    acc=__builtin_elementwise_max(acc,val & (u16x8)0x7fff);
  }
  unsigned val=0;
  #pragma unroll
  for (int i=0;i<8;i++) val=max(val,(unsigned)acc[i]);
  val=block_max(val);
  if(threadIdx.x==0)out[blockIdx.x]=val;
}
#else
#include "kittens.cuh"
using namespace kittens;
// Each warp independently reduces the small partial array, avoiding a workgroup barrier.
extern "C" __global__ __launch_bounds__(THREADS) void fa_v_quantize(unsigned char* out, float* scale, const unsigned short* x, const unsigned* partial) {
  constexpr int max_length = GROUPS < 64 ? 64 : GROUPS;
  rv_naive<float,max_length> maxima;
  #pragma unroll
  for(int i=0;i<max_length/64;i++) {
    int idx=kittens::laneid()*(max_length/64)+i;
    maxima[0][i]=idx<GROUPS ? __uint_as_float(partial[idx]<<16) : 0.f;
  }
  float val;
  kittens::max(val,maxima);
  float descale=(val+1.e-8f)*(1.f/448.f);
  if(blockIdx.x==0 && threadIdx.x==0)scale[0]=descale;
  float inv=__builtin_amdgcn_rcpf(descale);
  gl<bf16,-1,-1,-1,-1> input{(bf16*)x,1,1,1,SIZE};
  #pragma unroll
  for(int j=0;j<TILE/(THREADS*8);j++) {
    int tile=blockIdx.x*(TILE/512)+j*(THREADS/64)+threadIdx.x/64;
    rv_naive<float,512> vals;
    kittens::load(vals,input,{0,0,0,tile});
    kittens::mul(vals,vals,inv);
    kittens::max(vals,vals,-448.f);
    kittens::min(vals,vals,448.f);
    u32x2 result;
    #pragma unroll
    for(int i=0;i<2;i++) {
      unsigned lo=__builtin_amdgcn_cvt_pk_fp8_f32(vals[0][4*i],vals[0][4*i+1],0,false);
      result[i]=__builtin_amdgcn_cvt_pk_fp8_f32(vals[0][4*i+2],vals[0][4*i+3],lo,true);
    }
    *(u32x2*)(out+tile*512+kittens::laneid()*8)=result;
  }
}
#endif
