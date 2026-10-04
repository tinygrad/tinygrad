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
#define D_DIM 3072
#endif
#ifndef K_DIM
#define K_DIM 4
#endif
#ifndef THREADS
#define THREADS 64
#endif

static_assert(K_DIM == 4, "the GPTOSS gather-sum specialization requires top-k 4");
static_assert(D_DIM % THREADS == 0, "hidden width must divide evenly across the workgroup");
static_assert(THREADS % 64 == 0, "workgroup must contain whole AMD wavefronts");

extern "C" __global__ __launch_bounds__(THREADS) void
ggather_sum_fwd(__hip_bfloat16* __restrict__ out,
                const __hip_bfloat16* __restrict__ table,
                const int* __restrict__ idx) {
  const int gt = blockIdx.x;
  if (gt >= G_DIM * T_DIM) return;
  const int g = gt / T_DIM;
  const int t = gt - g * T_DIM;

  // Every lane uses the same naturally-aligned int4. LLVM lowers this uniform address without the generalized
  // modulo arithmetic of the UOp fallback; avoiding an LDS round trip also keeps the one-wave launch barrier-free.
  const int4 rowv = *reinterpret_cast<const int4*>(&idx[(long long)gt * K_DIM]);
  const int rowv0 = rowv.x, rowv1 = rowv.y, rowv2 = rowv.z, rowv3 = rowv.w;

  const long long group_base = (long long)g * M_DIM * D_DIM;
  const long long row0 = group_base + (long long)rowv0 * D_DIM;
  const long long row1 = group_base + (long long)rowv1 * D_DIM;
  const long long row2 = group_base + (long long)rowv2 * D_DIM;
  const long long row3 = group_base + (long long)rowv3 * D_DIM;
  const long long out_base = (long long)gt * D_DIM;

  {
#pragma clang fp reassociate(off)
#pragma unroll 2
    for (int d = threadIdx.x; d < D_DIM; d += THREADS) {
      float sum = (float)table[row0 + d];
      sum = sum + (float)table[row1 + d];
      sum = sum + (float)table[row2 + d];
      sum = sum + (float)table[row3 + d];
      out[out_base + d] = (__hip_bfloat16)sum;
    }
  }
}
