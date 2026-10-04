#include <hip/hip_runtime.h>

#ifndef G_DIM
#define G_DIM 1
#endif
#ifndef M_DIM
#define M_DIM 73728
#endif
#ifndef MK_DIM
#define MK_DIM 65536
#endif
#ifndef E_DIM
#define E_DIM 32
#endif
#ifndef THREADS
#define THREADS 256
#endif

// Build grouped-row -> expanded-source in one launch. Valid expert rows are filled by the inverse scatter; padding
// rows are filled with -1 from counts/off. These writes are disjoint, so no grid-wide initialization barrier is needed.
extern "C" __global__ __launch_bounds__(THREADS) void moe_invmap_padded(
    int *__restrict__ src_row,
    const int *__restrict__ dest_row,
    const int *__restrict__ counts,
    const int *__restrict__ off) {
  const int group = blockIdx.x / ((M_DIM + THREADS - 1) / THREADS);
  const int i = (blockIdx.x % ((M_DIM + THREADS - 1) / THREADS)) * THREADS + threadIdx.x;
  if (group >= G_DIM) return;

  if (i < MK_DIM) src_row[(long long)group * M_DIM + dest_row[(long long)group * MK_DIM + i]] = i;
  if (i < M_DIM) {
    int expert = 0;
    #pragma unroll
    for (int e = 1; e < E_DIM; e++) expert += i >= off[group * (E_DIM + 1) + e];
    if (i >= off[group * (E_DIM + 1) + expert] + counts[group * E_DIM + expert])
      src_row[(long long)group * M_DIM + i] = -1;
  }
}
