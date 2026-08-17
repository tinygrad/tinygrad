#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace mxfp4 {

struct Quantized4 {
  uint16_t fp4;
  uint8_t scale;
};

__device__ __forceinline__ float swizzle_xor1(float value) {
  float result;
  asm volatile("ds_swizzle_b32 %0, %1 offset:0x041f\n\ts_waitcnt lgkmcnt(0)" : "=v"(result) : "v"(value));
  return result;
}

__device__ __forceinline__ float swizzle_xor2(float value) {
  float result;
  asm volatile("ds_swizzle_b32 %0, %1 offset:0x081f\n\ts_waitcnt lgkmcnt(0)" : "=v"(result) : "v"(value));
  return result;
}

__device__ __forceinline__ float swizzle_xor4(float value) {
  float result;
  asm volatile("ds_swizzle_b32 %0, %1 offset:0x101f\n\ts_waitcnt lgkmcnt(0)" : "=v"(result) : "v"(value));
  return result;
}

__device__ __forceinline__ float max8(float value) {
  value = fmaxf(value, swizzle_xor4(value));
  value = fmaxf(value, swizzle_xor2(value));
  return fmaxf(value, swizzle_xor1(value));
}

__device__ __forceinline__ float4 load_bf16x4(const uint16_t* values) {
  const uint32_t lo = *reinterpret_cast<const uint32_t*>(values);
  const uint32_t hi = *reinterpret_cast<const uint32_t*>(values + 2);
  return make_float4(__uint_as_float(lo << 16), __uint_as_float(lo & 0xffff0000u),
                     __uint_as_float(hi << 16), __uint_as_float(hi & 0xffff0000u));
}

__device__ __forceinline__ void hadamard16(float4& value, int lane) {
  const float a0 = value.x + value.y, a1 = value.x - value.y;
  const float a2 = value.z + value.w, a3 = value.z - value.w;
  value = make_float4(a0 + a2, a1 + a3, a0 - a2, a1 - a3);

  const float4 xor1 = make_float4(swizzle_xor1(value.x), swizzle_xor1(value.y), swizzle_xor1(value.z), swizzle_xor1(value.w));
  value = lane & 1 ? make_float4(xor1.x - value.x, xor1.y - value.y, xor1.z - value.z, xor1.w - value.w)
                   : make_float4(xor1.x + value.x, xor1.y + value.y, xor1.z + value.z, xor1.w + value.w);

  const float4 xor2 = make_float4(swizzle_xor2(value.x), swizzle_xor2(value.y), swizzle_xor2(value.z), swizzle_xor2(value.w));
  value = lane & 2 ? make_float4(xor2.x - value.x, xor2.y - value.y, xor2.z - value.z, xor2.w - value.w)
                   : make_float4(xor2.x + value.x, xor2.y + value.y, xor2.z + value.z, xor2.w + value.w);
  value.x *= 0.25f;
  value.y *= 0.25f;
  value.z *= 0.25f;
  value.w *= 0.25f;
}

__device__ __forceinline__ uint8_t e8m0_scale(float amax, float& scale) {
  if (amax == 0.0f) {
    scale = 1.0f;
    return 127;
  }
  const uint32_t rounded = (__float_as_uint(amax) + 0x200000u) & 0xff800000u;
  int exponent = static_cast<int>((rounded >> 23) & 0xff) - 129;
  exponent = exponent < -127 ? -127 : exponent > 127 ? 127 : exponent;
  scale = exponent == -127 ? __uint_as_float(0x00400000u) : __uint_as_float(static_cast<uint32_t>(exponent + 127) << 23);
  return static_cast<uint8_t>(exponent + 127);
}

__device__ __forceinline__ uint16_t pack_fp4(float4 value, float scale) {
  uint32_t lo = 0, hi = 0;
  asm volatile("v_cvt_scalef32_pk_fp4_f32 %0, %1, %2, %3" : "+v"(lo) : "v"(value.x), "v"(value.y), "v"(scale));
  asm volatile("v_cvt_scalef32_pk_fp4_f32 %0, %1, %2, %3" : "+v"(hi) : "v"(value.z), "v"(value.w), "v"(scale));
  return static_cast<uint16_t>(lo | (hi << 8));
}

__device__ __forceinline__ Quantized4 quantize(float4 value, int lane) {
  hadamard16(value, lane);
  const float local_max = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)), fmaxf(fabsf(value.z), fabsf(value.w)));
  float scale;
  const uint8_t e8m0 = e8m0_scale(max8(local_max), scale);
  return {pack_fp4(value, scale), e8m0};
}

__device__ __forceinline__ void store_scale(uint8_t* output, int row, int col, int cols, uint8_t value) {
  const int tile = ((row >> 5) * (cols >> 3) + (col >> 3)) << 8;
  const int offset = ((col & 3) << 6) + ((row & 15) << 2) + (((col >> 2) & 1) << 1) + ((row >> 4) & 1);
  output[tile + offset] = value;
}

template<bool Shuffled>
__device__ __forceinline__ void store_fp4(uint8_t* output, int row, int col, int packed_cols, uint16_t value) {
  int index = row * packed_cols + col;
  if constexpr (Shuffled) {
    const int tile = (row >> 4) * (packed_cols << 4) + (col >> 5) * 512;
    const int offset = ((col >> 4) & 1) * 256 + (row & 15) * 16 + (col & 15);
    index = tile + offset;
  }
  *reinterpret_cast<uint16_t*>(output + index) = value;
}

} // namespace mxfp4
