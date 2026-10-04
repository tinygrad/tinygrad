#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#ifndef VOCAB
#define VOCAB 128256
#endif
#ifndef HIDDEN
#define HIDDEN 3072
#endif
#ifndef PART_M
#define PART_M 2
#endif
extern "C" __global__ __launch_bounds__(256) void lmhead_ce_reduce_dw(__hip_bfloat16 *out, const float *partials) {
  const long long idx = (long long)__builtin_amdgcn_workgroup_id_x() * 256 + __builtin_amdgcn_workitem_id_x();
  constexpr long long TOTAL = (long long)VOCAB * HIDDEN;
  if (idx >= TOTAL) return;
  float v = 0.0f;
  #pragma unroll
  for (int p = 0; p < PART_M; p++) v += partials[(long long)p * TOTAL + idx];
  out[idx] = (__hip_bfloat16)v;
}
