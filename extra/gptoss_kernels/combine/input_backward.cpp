#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

extern "C" __global__ __launch_bounds__(256) void gptoss_combine_input_backward(
    __hip_bfloat16 *__restrict__ out, const __hip_bfloat16 *__restrict__ dout,
    const int *__restrict__ src_row, const float *__restrict__ weights) {
  const int row = blockIdx.x, group = row / ROWS, tid = threadIdx.x;
  const int src = src_row[row];
  // Return before forming input addresses for unused grouped rows.
  if (src < 0) {
    for (int d = tid; d < 2880; d += 256) out[(long long)row * 2880 + d] = (__hip_bfloat16)0.0f;
    return;
  }

  const int token = group * TOKENS + src / 4;
  const float weight = (float)(__hip_bfloat16)weights[token * 4 + src % 4];
  for (int d = tid; d < 2880; d += 256) {
    const float value = (float)(__hip_bfloat16)((float)dout[(long long)token * 2880 + d] * weight);
    // The stock scatter adds to positive zero, including when the product is negative zero.
    const __hip_bfloat16 result = (__hip_bfloat16)(0.0f + value);
    out[(long long)row * 2880 + d] = (float)result == 0.0f ? (__hip_bfloat16)0.0f : result;
  }
}
