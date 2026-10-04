#include <hip/hip_runtime.h>

// One workgroup reads one local/received shard. Each lane moves sixteen raw FP8 bytes;
// no dequantization, rounding, or float loads. Physical row/column padding is initialized on every invocation.
constexpr int COL_VECTORS = 3072 / 16, LIVE_COL_VECTORS = 2880 / 16;
constexpr int VECTORS_PER_SHARD = 4 * PHYSICAL_ROWS * COL_VECTORS;
static_assert(VECTORS_PER_SHARD % 256 == 0);

extern "C" __global__ __launch_bounds__(256) void gptoss_compact_weight_gather(
    uint4 *__restrict__ out, const uint4 *__restrict__ p0, const uint4 *__restrict__ p1,
    const uint4 *__restrict__ p2, const uint4 *__restrict__ p3, const uint4 *__restrict__ p4,
    const uint4 *__restrict__ p5, const uint4 *__restrict__ p6, const uint4 *__restrict__ p7) {
  const int shard = __builtin_amdgcn_workgroup_id_y();
  const uint4 *src;
  switch (shard) {
    case 0: src = p0; break;
    case 1: src = p1; break;
    case 2: src = p2; break;
    case 3: src = p3; break;
    case 4: src = p4; break;
    case 5: src = p5; break;
    case 6: src = p6; break;
    default: src = p7;
  }
  const int idx = __builtin_amdgcn_workgroup_id_x() * 256 + __builtin_amdgcn_workitem_id_x();
  const int col = idx % COL_VECTORS, row = idx / COL_VECTORS;
  const int expert = row / PHYSICAL_ROWS, expert_row = row % PHYSICAL_ROWS;
  uint4 value = {0, 0, 0, 0};
  if (expert_row < LIVE_ROWS && col < LIVE_COL_VECTORS)
    value = src[(expert * LIVE_ROWS + expert_row) * LIVE_COL_VECTORS + col];
  out[shard * VECTORS_PER_SHARD + idx] = value;
}
