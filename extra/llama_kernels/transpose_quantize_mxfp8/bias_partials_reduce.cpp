#include <hip/hip_runtime.h>
#ifndef PARTIAL_ROWS
#define PARTIAL_ROWS 2304
#endif
#ifndef N_COLS
#define N_COLS 5888
#endif
#ifndef N_EXPERTS
#define N_EXPERTS 32
#endif
#ifndef REDUCE_UNROLL
#define REDUCE_UNROLL 1
#endif
#if REDUCE_UNROLL > 1
__device__ __forceinline__ float ordered_add(float a, float b) {
  float ret;
  asm volatile("v_add_f32 %0, %1, %2" : "=v"(ret) : "v"(a), "v"(b));
  return ret;
}
#endif
extern "C" __global__ __launch_bounds__(256) void bias_partials_reduce(
    float *__restrict__ out, const float *__restrict__ partial, const int *__restrict__ expert_off) {
  const int e=blockIdx.x/(N_COLS/256), n=(blockIdx.x%(N_COLS/256))*256+threadIdx.x;
  const int p0=expert_off[e]/32, p1=expert_off[e+1]/32;
#if REDUCE_UNROLL > 1
  static_assert(REDUCE_UNROLL % 2 == 0);
  // Clang implements the reference loop as packed two-way accumulation (even p, odd p), then one horizontal add.
  // Preserve that precise FP32 order while preloading a larger group to expose memory-level parallelism.
  float sum_even=0.0f, sum_odd=0.0f;
  int p=p0;
  for(;p+REDUCE_UNROLL<=p1;p+=REDUCE_UNROLL) {
    float vals[REDUCE_UNROLL];
#pragma unroll
    for(int u=0;u<REDUCE_UNROLL;u++) vals[u] = partial[(long long)(p+u)*N_COLS+n];
#pragma unroll
    for(int u=0;u<REDUCE_UNROLL;u+=2) {
      sum_even=ordered_add(sum_even, vals[u]);
      sum_odd=ordered_add(sum_odd, vals[u+1]);
    }
  }
  for(;p+1<p1;p+=2) {
    const float v0=partial[(long long)p*N_COLS+n], v1=partial[(long long)(p+1)*N_COLS+n];
    sum_even=ordered_add(sum_even, v0);
    sum_odd=ordered_add(sum_odd, v1);
  }
  float sum=ordered_add(sum_even, sum_odd);
  if(p<p1) sum=ordered_add(sum, partial[(long long)p*N_COLS+n]);
#else
  float sum=0.0f;
#pragma unroll 1
  for(int p=p0;p<p1;p++) sum += partial[(long long)p*N_COLS+n];
#endif
  out[e*N_COLS+n]=sum;
}
