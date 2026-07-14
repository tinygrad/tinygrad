#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#ifndef ATTN_B
#define ATTN_B 2
#endif
#ifndef ATTN_N
#define ATTN_N 8192
#endif
#ifndef ATTN_H
#define ATTN_H 32
#endif
#ifndef ATTN_H_KV
#define ATTN_H_KV 8
#endif
#ifndef ATTN_D
#define ATTN_D 128
#endif
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV;
constexpr int HALF_D = ATTN_D / 2;
constexpr int PACKED_D = (GROUP_SIZE + 2) * ATTN_D;

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK) void
fused_qkv_rope_backward(
    __hip_bfloat16*       __restrict__ dxqkv,
    const __hip_bfloat16* __restrict__ dq,
    const __hip_bfloat16* __restrict__ dk,
    const __hip_bfloat16* __restrict__ dv,
    const __hip_bfloat16* __restrict__ freqs_cis) {
  const int b = blockIdx.x;
  const int n = blockIdx.y;
  const int bn = b * ATTN_N + n;
  const int packed_bn = bn * ATTN_H_KV * PACKED_D;
  const int q_bn = bn * ATTN_H * ATTN_D;
  const int kv_bn = bn * ATTN_H_KV * ATTN_D;

  if (threadIdx.x < HALF_D) {
    const int pair = threadIdx.x;
    const int even = pair << 1;
    const float c = static_cast<float>(freqs_cis[((n * HALF_D + pair) * 2) + 0]);
    const float s = static_cast<float>(freqs_cis[((n * HALF_D + pair) * 2) + 1]);

    for (int kvh = 0; kvh < ATTN_H_KV; kvh++) {
      const int base = packed_bn + kvh * PACKED_D;

      for (int rep = 0; rep < GROUP_SIZE; rep++) {
        const int h = kvh * GROUP_SIZE + rep;
        const int in = q_bn + h * ATTN_D + even;
        const float ge = static_cast<float>(dq[in]);
        const float go = static_cast<float>(dq[in + 1]);
        const int qbase = base + rep * ATTN_D;
        dxqkv[qbase + even] = static_cast<__hip_bfloat16>(ge * c + go * s);
        dxqkv[qbase + even + 1] = static_cast<__hip_bfloat16>(-ge * s + go * c);
      }

      const int in = kv_bn + kvh * ATTN_D + even;
      const float ge = static_cast<float>(dk[in]);
      const float go = static_cast<float>(dk[in + 1]);
      dxqkv[base + GROUP_SIZE * ATTN_D + even] = static_cast<__hip_bfloat16>(ge * c + go * s);
      dxqkv[base + GROUP_SIZE * ATTN_D + even + 1] = static_cast<__hip_bfloat16>(-ge * s + go * c);
      dxqkv[base + (GROUP_SIZE + 1) * ATTN_D + even] = dv[in];
      dxqkv[base + (GROUP_SIZE + 1) * ATTN_D + even + 1] = dv[in + 1];
    }
  }
}
