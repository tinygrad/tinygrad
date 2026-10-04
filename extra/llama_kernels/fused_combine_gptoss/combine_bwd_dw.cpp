#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#ifndef G_DIM
#define G_DIM 1
#endif
#ifndef T_DIM
#define T_DIM 16384
#endif
#ifndef M_DIM
#define M_DIM 73728
#endif
#ifndef D_DIM
#define D_DIM 2880
#endif
#ifndef P_DIM
#define P_DIM D_DIM
#endif
#ifndef K_DIM
#define K_DIM 4
#endif
#ifndef THREADS_PER_WG
#define THREADS_PER_WG 256
#endif
#ifndef PIPE2_NTZ
#define PIPE2_NTZ 0
#endif

// d_weights[g,t,j] = dot(d_out[g,t,:], z[g,dest_row[g,t*K+j],:]).
// One workgroup owns all K router slots for a token: d_out is read once, both input
// streams are coalesced, and only the K final fp32 sums reach HBM. This replaces the
// gathered-z materialization plus its generic reduction.
constexpr int WARP_SIZE = 64;
static_assert(THREADS_PER_WG % WARP_SIZE == 0);
static_assert(K_DIM > 0);
static_assert(D_DIM <= P_DIM);

__device__ __forceinline__ __hip_bfloat16 load_z(const __hip_bfloat16* ptr) {
#if PIPE2_NTZ
  // z is a one-use 432 MiB backward checkpoint at the exact GPTOSS local shape.
  // Bypass temporal caching so its four gathered streams do not displace d_out.
  const unsigned short raw = __builtin_nontemporal_load(reinterpret_cast<const unsigned short*>(ptr));
  return __builtin_bit_cast(__hip_bfloat16, raw);
#else
  return *ptr;
#endif
}

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
combine_bwd_dw(float* __restrict__ dw,
               const __hip_bfloat16* __restrict__ z,
               const __hip_bfloat16* __restrict__ d_out,
               const int* __restrict__ dest_row) {
  const int gt = blockIdx.x;
  const int g = gt / T_DIM;
  const int t = gt - g * T_DIM;
  if (g >= G_DIM) return;

  const int tid = threadIdx.x;
  const long long dout_base = (long long)gt * D_DIM;
  const long long dest_base = (long long)gt * K_DIM;
  float sums[K_DIM] = {};
  int rows[K_DIM];

#pragma unroll
  for (int j = 0; j < K_DIM; j++) rows[j] = dest_row[dest_base + j];

#if PIPE2_NTZ
  static_assert(D_DIM == 2880 && THREADS_PER_WG == 256,
                "the pipelined tail is specialized to the GPTOSS production width");
  // Every lane owns eleven elements; lanes 0:63 own a twelfth. Pairing the first ten
  // overlaps global loads while retaining each route's exact d=tid+256*i fmaf chain.
#pragma unroll
  for (int i = 0; i < 5; i++) {
    const int d0 = tid + i * (2 * THREADS_PER_WG);
    const int d1 = d0 + THREADS_PER_WG;
    const float dy0 = (float)d_out[dout_base + d0];
    const float dy1 = (float)d_out[dout_base + d1];
#pragma unroll
    for (int j = 0; j < K_DIM; j++) {
      const float z0 = (float)load_z(&z[((long long)g * M_DIM + rows[j]) * P_DIM + d0]);
      const float z1 = (float)load_z(&z[((long long)g * M_DIM + rows[j]) * P_DIM + d1]);
      sums[j] = fmaf(dy0, z0, sums[j]);
      sums[j] = fmaf(dy1, z1, sums[j]);
    }
  }
  int d = tid + 10 * THREADS_PER_WG;
  float dy = (float)d_out[dout_base + d];
#pragma unroll
  for (int j = 0; j < K_DIM; j++)
    sums[j] = fmaf(dy, (float)load_z(&z[((long long)g * M_DIM + rows[j]) * P_DIM + d]), sums[j]);
  if (tid < D_DIM - 11 * THREADS_PER_WG) {
    d += THREADS_PER_WG;
    dy = (float)d_out[dout_base + d];
#pragma unroll
    for (int j = 0; j < K_DIM; j++)
      sums[j] = fmaf(dy, (float)load_z(&z[((long long)g * M_DIM + rows[j]) * P_DIM + d]), sums[j]);
  }
#else
  for (int d = tid; d < D_DIM; d += THREADS_PER_WG) {
    const float dy = (float)d_out[dout_base + d];
#pragma unroll
    for (int j = 0; j < K_DIM; j++)
      sums[j] = fmaf(dy, (float)load_z(&z[((long long)g * M_DIM + rows[j]) * P_DIM + d]), sums[j]);
  }
#endif

#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
#pragma unroll
    for (int j = 0; j < K_DIM; j++) sums[j] += __shfl_down(sums[j], offset, WARP_SIZE);

  constexpr int NWARPS = THREADS_PER_WG / WARP_SIZE;
  __shared__ float warp_sums[NWARPS][K_DIM];
  const int lane = tid % WARP_SIZE;
  const int warp = tid / WARP_SIZE;
#pragma unroll
  for (int j = 0; j < K_DIM; j++) if (lane == 0) warp_sums[warp][j] = sums[j];
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int j = 0; j < K_DIM; j++) sums[j] = lane < NWARPS ? warp_sums[lane][j] : 0.0f;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
#pragma unroll
      for (int j = 0; j < K_DIM; j++) sums[j] += __shfl_down(sums[j], offset, WARP_SIZE);
    if (lane == 0)
#pragma unroll
      for (int j = 0; j < K_DIM; j++) dw[dest_base + j] = sums[j];
  }
}
