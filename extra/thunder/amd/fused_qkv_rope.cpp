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
fused_qkv_rope_forward(
    __hip_bfloat16*       __restrict__ q,
    __hip_bfloat16*       __restrict__ k,
    __hip_bfloat16*       __restrict__ v,
    const __hip_bfloat16* __restrict__ xqkv,
    const __hip_bfloat16* __restrict__ freqs_cis) {
  const int b = blockIdx.x;
  const int n = blockIdx.y;
  const int bn = b * ATTN_N + n;
  const int packed_bn = bn * ATTN_H_KV * PACKED_D;
  const int q_bn = bn * ATTN_H * ATTN_D;
  const int kv_bn = bn * ATTN_H_KV * ATTN_D;

  // Distribute all (kv-head, slot, pair) work across ALL threads (the old code used only HALF_D threads -> at D=64
  // that was 32/256 = 12.5% occupancy). Each kv-head has (GROUP_SIZE q-heads + 1 k + 1 v) slots, each HALF_D pairs.
  constexpr int SLOTS_PER_KV = GROUP_SIZE + 2;
  constexpr int TOTAL = ATTN_H_KV * SLOTS_PER_KV * HALF_D;
  for (int idx = threadIdx.x; idx < TOTAL; idx += THREADS_PER_BLOCK) {
    const int pair = idx % HALF_D;
    const int slot = idx / HALF_D;
    const int kvh = slot / SLOTS_PER_KV;
    const int slot_in_kv = slot % SLOTS_PER_KV;
    const int even = pair << 1;
    const int in_base = packed_bn + kvh * PACKED_D + slot_in_kv * ATTN_D + even;
    if (slot_in_kv == GROUP_SIZE + 1) {                        // v: straight copy (no rope)
      const int out = kv_bn + kvh * ATTN_D + even;
      v[out]     = xqkv[in_base];
      v[out + 1] = xqkv[in_base + 1];
    } else {                                                   // q or k: rope rotation
      const float a  = static_cast<float>(xqkv[in_base]);
      const float bb = static_cast<float>(xqkv[in_base + 1]);
      const float c  = static_cast<float>(freqs_cis[((n * HALF_D + pair) * 2) + 0]);
      const float s  = static_cast<float>(freqs_cis[((n * HALF_D + pair) * 2) + 1]);
      int out;
      if (slot_in_kv < GROUP_SIZE) out = q_bn + (kvh * GROUP_SIZE + slot_in_kv) * ATTN_D + even;
      else                         out = kv_bn + kvh * ATTN_D + even;                    // k
      __hip_bfloat16* dst = (slot_in_kv < GROUP_SIZE) ? q : k;
      dst[out]     = static_cast<__hip_bfloat16>(a * c - bb * s);
      dst[out + 1] = static_cast<__hip_bfloat16>(a * s + bb * c);
    }
  }
}
