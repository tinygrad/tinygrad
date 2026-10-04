#include <hip/hip_runtime.h>

// Fixed GPT-OSS attention ABI: 16384x4096 BF16, row FP8/E8M0, and mx_pack's uint32 layout.
typedef unsigned short u16;
typedef unsigned int u32;
typedef u16 input_vec __attribute__((ext_vector_type(16)));
typedef unsigned char output_vec __attribute__((ext_vector_type(16)));

extern "C" __global__ __launch_bounds__(128) void gptoss_attention_quantize(
    unsigned char* __restrict__ q, unsigned char* __restrict__ e8, u32* __restrict__ si,
    const u16* __restrict__ x) {
  int thread = blockIdx.x*128+threadIdx.x;
  int block = thread/2, lane = thread%2;
  input_vec bits = *reinterpret_cast<const input_vec*>(x+thread*16);
  int exponent = 0;
  #pragma unroll
  for (int k=0; k<16; k++) {
    int abs_bits = bits[k]&0x7fff;
    exponent = max(exponent,abs_bits>0x7f80 ? 0 : abs_bits>>7);
  }
  exponent = max(exponent,__shfl_xor(exponent,1,2));
  // Match the generated serial maximum's NaN behavior, including a NaN at block offset zero.
  if (__shfl(int(bits[0]&0x7fff),0,2)>0x7f80) exponent = 255;
  exponent = min(exponent,254);
  float scale = __builtin_bit_cast(float,u32(254-exponent)<<23);
  output_vec quant;
  #pragma unroll
  for (int k=0; k<16; k+=2) {
    float a = __builtin_bit_cast(float,u32(bits[k])<<16)*scale;
    float b = __builtin_bit_cast(float,u32(bits[k+1])<<16)*scale;
    a = a < -448.f ? -448.f : a;
    a = a > 448.f ? 448.f : a;
    b = b < -448.f ? -448.f : b;
    b = b > 448.f ? 448.f : b;
    u32 value = __builtin_amdgcn_cvt_pk_fp8_f32(a,b,0,false);
    quant[k] = value;
    quant[k+1] = value>>8;
  }
  *reinterpret_cast<output_vec*>(q+thread*16) = quant;
  if (!lane) {
    e8[block] = exponent;
    // Distinct byte stores, no RMW: four adjacent MX blocks fill one scale-major uint32 word.
    reinterpret_cast<unsigned char*>(si)[((block%128)/4*16384+block/128)*4+block%4] = exponent;
  }
}
