#include <hip/hip_runtime.h>

// GPT-OSS activation q/e8 only. Two adjacent lanes own a 32-element MX block.
// The arithmetic matches the generated FAST_E8M0 path, including its ordered NaN behavior.
typedef unsigned short input_vec __attribute__((ext_vector_type(16)));
typedef unsigned char output_vec __attribute__((ext_vector_type(16)));
static_assert(THREADS == 128 || THREADS == 256);

extern "C" __global__ __launch_bounds__(THREADS) void quantize_mxfp8_qe8_vector(
    unsigned char* __restrict__ q, unsigned char* __restrict__ e8, const unsigned short* __restrict__ x) {
  const int thread = blockIdx.x*THREADS + threadIdx.x;
  const input_vec bits = *reinterpret_cast<const input_vec*>(x + thread*16);
  int exponent = 0;
#pragma unroll
  for (int k = 0; k < 16; k++) {
    const int magnitude = bits[k] & 0x7fff;
    exponent = max(exponent, magnitude > 0x7f80 ? 0 : magnitude >> 7);
  }
  exponent = max(exponent, __shfl_xor(exponent, 1, 2));
  if (__shfl(int(bits[0] & 0x7fff), 0, 2) > 0x7f80) exponent = 255;
  exponent = min(exponent, 254);
  const float scale = __builtin_bit_cast(float, unsigned(254-exponent) << 23);
  output_vec packed;
#pragma unroll
  for (int k = 0; k < 16; k += 2) {
    float a = __builtin_bit_cast(float, unsigned(bits[k]) << 16) * scale;
    float b = __builtin_bit_cast(float, unsigned(bits[k+1]) << 16) * scale;
    a = a < -448.f ? -448.f : a; a = a > 448.f ? 448.f : a;
    b = b < -448.f ? -448.f : b; b = b > 448.f ? 448.f : b;
    const unsigned value = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    packed[k] = value; packed[k+1] = value >> 8;
  }
  *reinterpret_cast<output_vec*>(q + thread*16) = packed;
  if (!(thread & 1)) e8[thread/2] = exponent;
}
