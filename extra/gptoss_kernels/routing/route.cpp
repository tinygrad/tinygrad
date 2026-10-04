#include <hip/hip_runtime.h>

#ifndef ROUTE_PHASE
#error ROUTE_PHASE is required
#endif

constexpr int ENTRIES = 65536;
constexpr int EXPERTS = 32;
constexpr int BLOCK = 256;
constexpr int BLOCKS = ENTRIES / BLOCK;
constexpr int PADDED = ENTRIES + EXPERTS * BLOCK;

#if ROUTE_PHASE == 0
// Preserve the original flattened top-k entry order, including repeated IDs.
extern "C" __global__ __launch_bounds__(256) void route_histogram(
    int *__restrict__ local_rank, int *__restrict__ histogram, const int *__restrict__ topi) {
  __shared__ int wave_counts[4][EXPERTS];
  const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
  const int expert = topi[blockIdx.x * BLOCK + tid];
  const unsigned long long earlier = (1ull << lane) - 1;
  int rank = 0;
  #pragma unroll
  for (int e = 0; e < EXPERTS; ++e) {
    const unsigned long long selected = __ballot(expert == e);
    if (expert == e) rank = __popcll(selected & earlier);
    if (lane == 0) wave_counts[wave][e] = __popcll(selected);
  }
  __syncthreads();
  #pragma unroll
  for (int w = 0; w < 4; ++w) if (w < wave) rank += wave_counts[w][expert];
  local_rank[blockIdx.x * BLOCK + tid] = rank;
  if (tid < EXPERTS) {
    int count = 0;
    #pragma unroll
    for (int w = 0; w < 4; ++w) count += wave_counts[w][tid];
    histogram[blockIdx.x * EXPERTS + tid] = count;
  }
}
#elif ROUTE_PHASE == 1
// A single workgroup owns the small inter-block scan and expert offsets, so
// no atomics, grid-wide barriers or uninitialized counter state are needed.
extern "C" __global__ __launch_bounds__(1024) void route_scan(
    int *__restrict__ block_prefix, int *__restrict__ counts, int *__restrict__ off,
    const int *__restrict__ histogram) {
  __shared__ int totals[EXPERTS];
  const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
  #pragma unroll
  for (int j = 0; j < EXPERTS / 16; ++j) {
    const int expert = wave * (EXPERTS / 16) + j;
    int values[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) values[k] = histogram[(lane * 4 + k) * EXPERTS + expert];
    const int subtotal = values[0] + values[1] + values[2] + values[3];
    int inclusive = subtotal;
    #pragma unroll
    for (int delta = 1; delta < 64; delta *= 2) {
      const int other = __shfl_up(inclusive, delta, 64);
      if (lane >= delta) inclusive += other;
    }
    int prefix = inclusive - subtotal;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      block_prefix[(lane * 4 + k) * EXPERTS + expert] = prefix;
      prefix += values[k];
    }
    if (lane == 63) totals[expert] = inclusive;
  }
  __syncthreads();
  if (wave == 0) {
    const int count = lane < EXPERTS ? totals[lane] : 0;
    const int padded = ((count + BLOCK - 1) / BLOCK) * BLOCK;
    int inclusive = padded;
    #pragma unroll
    for (int delta = 1; delta < 64; delta *= 2) {
      const int other = __shfl_up(inclusive, delta, 64);
      if (lane >= delta) inclusive += other;
    }
    if (lane < EXPERTS) {
      counts[lane] = count;
      off[lane] = inclusive - padded;
    }
    if (lane == EXPERTS - 1) off[EXPERTS] = inclusive;
  }
}
#elif ROUTE_PHASE == 2
extern "C" __global__ __launch_bounds__(256) void route_scatter(
    int *__restrict__ dest_row, int *__restrict__ src_row, const int *__restrict__ topi,
    const int *__restrict__ local_rank, const int *__restrict__ block_prefix,
    const int *__restrict__ counts, const int *__restrict__ off) {
  __shared__ int offsets[EXPERTS + 1], lengths[EXPERTS];
  const int tid = threadIdx.x;
  if (tid <= EXPERTS) offsets[tid] = off[tid];
  if (tid < EXPERTS) lengths[tid] = counts[tid];
  __syncthreads();
  const int i = blockIdx.x * BLOCK + tid;
  if (i < ENTRIES) {
    const int expert = topi[i];
    const int destination = offsets[expert] + block_prefix[(i / BLOCK) * EXPERTS + expert] + local_rank[i];
    dest_row[i] = destination;
    src_row[destination] = i;
  }
  if (i < PADDED) {
    // Duplicate offsets from empty experts must select the last equal start.
    int expert = 0;
    #pragma unroll
    for (int e = 1; e < EXPERTS; ++e) expert += i >= offsets[e];
    if (i >= offsets[expert] + lengths[expert]) src_row[i] = -1;
  }
}
#else
#error unsupported ROUTE_PHASE
#endif
