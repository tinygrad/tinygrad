#include <hip/hip_runtime.h>

#ifndef G_DIM
#define G_DIM 1
#endif
#ifndef M_DIM
#define M_DIM 73728
#endif
#ifndef T_DIM
#define T_DIM 16384
#endif
#ifndef D_DIM
#define D_DIM 3072
#endif
#ifndef TOPK
#define TOPK 4
#endif
#ifndef THREADS
#define THREADS 256
#endif

// Gather raw row scales straight into mx_pack's scale-major representation. One thread owns one uint32 output,
// so both its four-byte load and store are naturally aligned and adjacent lanes write adjacent routed rows.
extern "C" __global__ __launch_bounds__(THREADS) void dispatch_row_si(
    uint32_t *__restrict__ row_si,
    const uint8_t *__restrict__ e8_in,
    const int *__restrict__ src_row) {
  constexpr long long ROWS = (long long)G_DIM * M_DIM;
  constexpr int SCALE_GROUPS = D_DIM / 128;
  const long long idx = (long long)blockIdx.x * THREADS + threadIdx.x;
  if (idx >= ROWS * SCALE_GROUPS) return;
  const int scale_group = idx / ROWS;
  const int grouped_row = idx - (long long)scale_group * ROWS;
  const int group = grouped_row / M_DIM;
  const int src = src_row[grouped_row];
  uint32_t packed = 0;
  if (src >= 0) {
    const int token = src / TOPK;
    packed = *reinterpret_cast<const uint32_t *>(
      &e8_in[((long long)group * T_DIM + token) * (D_DIM / 32) + scale_group * 4]);
  }
  row_si[idx] = packed;
}
