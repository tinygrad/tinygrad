// GPT-OSS: weighted dZ, both MX layouts, and ordered bias partials.
// Shared BF16 values retain the rounding boundary. Row blocks quantize locally;
// columns retain the baseline's sequential sum and coalesced transposed stores.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
constexpr int M=73728,N=3072,D=2880,TM=32,TN=256,STRIDE=264;
typedef short short2v __attribute__((ext_vector_type(2)));
__device__ __forceinline__ unsigned cvt4(float a,float b,float c,float d,int e) {
  const float sf=__builtin_bit_cast(float,(unsigned)e<<23);
  short2v lo={0,0},hi={0,0};
  lo=__builtin_amdgcn_cvt_scalef32_pk_fp8_f32(lo,a,b,sf,false);
  hi=__builtin_amdgcn_cvt_scalef32_pk_fp8_f32(hi,c,d,sf,false);
  return (__builtin_bit_cast(unsigned,lo)&65535u)|(__builtin_bit_cast(unsigned,hi)<<16);
}
extern "C" __global__ __launch_bounds__(256) void combine_bwd_dz_quant(
 unsigned char* __restrict__ row_q,unsigned char* __restrict__ row_e8,
 unsigned char* __restrict__ col_q,unsigned char* __restrict__ col_e8,
 float* __restrict__ bias_partial,const __hip_bfloat16* __restrict__ d_out,
 const int* __restrict__ src_row,const float* __restrict__ weights) {
  const int tid=threadIdx.x,tile_m=blockIdx.x/(N/TN),tile_n=blockIdx.x%(N/TN);
  __shared__ __hip_bfloat16 lds[TM*STRIDE];
  const int pair=(tid&127)*2,r0=(tid>>7)*16,n0=tile_n*TN+pair;
  #pragma unroll
  for (int rr=0;rr<16;rr++) {
    const int row=r0+rr,m=tile_m*TM+row;
    const int sr=__builtin_amdgcn_readfirstlane(src_row[m]);
    unsigned short a=0,b=0;
    if (sr>=0 && n0<D) {
      const unsigned packed=*reinterpret_cast<const unsigned*>(&d_out[(long long)(sr/4)*D+n0]);
      const float w=weights[sr];
      a=__builtin_bit_cast(unsigned short,(__hip_bfloat16)((float)__builtin_bit_cast(__hip_bfloat16,(unsigned short)packed)*w));
      b=__builtin_bit_cast(unsigned short,(__hip_bfloat16)((float)__builtin_bit_cast(__hip_bfloat16,(unsigned short)(packed>>16))*w));
    }
    *reinterpret_cast<unsigned*>(&lds[row*STRIDE+pair])=(unsigned)a|((unsigned)b<<16);
  }
  __syncthreads();
  const int row=tid/8,sub=tid%8;
  float values[32],ra=0.0f;
  #pragma unroll
  for(int k=0;k<32;k++) { values[k]=(float)lds[row*STRIDE+sub*32+k];ra=fmaxf(ra,fabsf(values[k])); }
  const int rs=min(254,(int)((__builtin_bit_cast(unsigned,ra)>>23)&255));
  row_e8[(long long)(tile_m*TM+row)*(N/32)+tile_n*8+sub]=rs;
  #pragma unroll
  for(int k=0;k<32;k+=4)
    *reinterpret_cast<unsigned*>(&row_q[(long long)(tile_m*TM+row)*N+tile_n*TN+sub*32+k])=
      cvt4(values[k],values[k+1],values[k+2],values[k+3],rs);
  float amax=0.0f,tile_sum=0.0f;
  #pragma unroll
  for(int mm=0;mm<TM;mm++) {
    const float v=(float)lds[mm*STRIDE+tid];amax=fmaxf(amax,fabsf(v));if (mm==0) tile_sum=v; else asm("v_add_f32 %0, %1, %2" : "=v"(tile_sum) : "v"(tile_sum), "v"(v));
  }
  const int n=tile_n*TN+tid,e8=min(254,(int)((__builtin_bit_cast(unsigned,amax)>>23)&255));
  col_e8[((long long)(tile_m/4)*N+n)*4+tile_m%4]=e8;
  bias_partial[(long long)tile_m*N+n]=tile_sum;
  const int lane=tid&63,wid=tid>>6,mm4=(lane&7)*4,nbase=wid*64+(lane>>3);
  #pragma unroll
  for(int j=0;j<8;j++) {
    const int ln=nbase+j*8,scale=__shfl(e8,(lane>>3)+j*8,64);
    *reinterpret_cast<unsigned*>(&col_q[(long long)(tile_n*TN+ln)*M+tile_m*TM+mm4])=
      cvt4((float)lds[(mm4+0)*STRIDE+ln],(float)lds[(mm4+1)*STRIDE+ln],
           (float)lds[(mm4+2)*STRIDE+ln],(float)lds[(mm4+3)*STRIDE+ln],scale);
  }
}
