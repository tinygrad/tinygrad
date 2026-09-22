#pragma once
#include "quantize_mxfp4_device.h"

namespace mxfp4 {
struct Quantized8 { uint32_t fp4; uint8_t scale; };
__device__ __forceinline__ float add_xor1(float x,float y) {
  return __uint_as_float(__builtin_amdgcn_mov_dpp(__float_as_uint(x),0xb1,0xf,0xf,false))+y;
}
__device__ __forceinline__ float flip(float x,uint32_t sign) { return __uint_as_float(__float_as_uint(x)^sign); }

// Four lanes own one 32-element scale block, producing aligned 32-bit FP4 stores.
__device__ __forceinline__ Quantized8 quantize8(float4 a,float4 b,int lane) {
  const float a0=a.x+a.y,a1=a.x-a.y,a2=a.z+a.w,a3=a.z-a.w;
  const float b0=b.x+b.y,b1=b.x-b.y,b2=b.z+b.w,b3=b.z-b.w;
  a=make_float4(a0+a2,a1+a3,a0-a2,a1-a3);
  b=make_float4(b0+b2,b1+b3,b0-b2,b1-b3);
  const float4 lo=make_float4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);
  b=make_float4(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w);
  a=lo;
  const uint32_t sign=uint32_t(lane&1)<<31;
  a=make_float4(add_xor1(a.x,flip(a.x,sign)),add_xor1(a.y,flip(a.y,sign)),
                add_xor1(a.z,flip(a.z,sign)),add_xor1(a.w,flip(a.w,sign)));
  b=make_float4(add_xor1(b.x,flip(b.x,sign)),add_xor1(b.y,flip(b.y,sign)),
                add_xor1(b.z,flip(b.z,sign)),add_xor1(b.w,flip(b.w,sign)));
  a.x*=0.25f; a.y*=0.25f; a.z*=0.25f; a.w*=0.25f;
  b.x*=0.25f; b.y*=0.25f; b.z*=0.25f; b.w*=0.25f;
  float mx=fmaxf(fmaxf(fmaxf(fabsf(a.x),fabsf(a.y)),fmaxf(fabsf(a.z),fabsf(a.w))),
                  fmaxf(fmaxf(fabsf(b.x),fabsf(b.y)),fmaxf(fabsf(b.z),fabsf(b.w))));
  mx=fmaxf(mx,__uint_as_float(__builtin_amdgcn_mov_dpp(__float_as_uint(mx),0xb1,0xf,0xf,false)));
  mx=fmaxf(mx,__uint_as_float(__builtin_amdgcn_mov_dpp(__float_as_uint(mx),0x4e,0xf,0xf,false)));
  float scale;
  const uint8_t e8=e8m0_scale(mx,scale);
  return {uint32_t(pack_fp4(a,scale))|(uint32_t(pack_fp4(b,scale))<<16),e8};
}
template<bool Shuffled>
__device__ __forceinline__ void store_fp4_8(uint8_t* output,int row,int col,int packed_cols,uint32_t value) {
  int index=row*packed_cols+col;
  if constexpr (Shuffled) index=(row>>4)*(packed_cols<<4)+(col>>5)*512+((col>>4)&1)*256+(row&15)*16+(col&15);
  *reinterpret_cast<uint32_t*>(output+index)=value;
}
} // namespace mxfp4
