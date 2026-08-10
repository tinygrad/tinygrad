#include "kittens.cuh"

using namespace kittens;

#ifndef MATVEC_N
#define MATVEC_N 1536
#endif
#ifndef MATVEC_K
#define MATVEC_K 7168
#endif

constexpr int SPLIT_WAVES = 8;

template<int W>
__device__ __forceinline__ float run_split(const bf16 *A_ptr, const bf16 *B_ptr, int out_base,
                                           st_bf<16, 32, st_16x32_s> &As,
                                           st_bf<16, 32, st_16x32_s> &Bs) {
  constexpr int K = MATVEC_K;
  rt_bf<16, 32, row_l, rt_16x32_s> A;
  rt_bf<16, 32, row_l, rt_16x32_s> B;
  rt_fl<16, 16, col_l, rt_16x16_s> C;
  zero(C);
  const int lane = laneid();
  constexpr int k_begin = W * (K / SPLIT_WAVES), k_end = k_begin + K / SPLIT_WAVES;
  #pragma unroll 1
  for (int k = k_begin; k < k_end; k += 32) {
    #pragma unroll
    for (int idx = lane; idx < 16 * 32; idx += 64) {
      const int row = idx / 32, col = idx % 32;
      *reinterpret_cast<bf16 *>(reinterpret_cast<char *>(&As.data[0]) + As.swizzle({row, col})) = A_ptr[k + col];
      *reinterpret_cast<bf16 *>(reinterpret_cast<char *>(&Bs.data[0]) + Bs.swizzle({row, col})) =
        B_ptr[(out_base + row) * K + k + col];
    }
    asm volatile("s_waitcnt lgkmcnt(0)");
    load(A, As);
    load(B, Bs);
    asm volatile("s_waitcnt lgkmcnt(0)");
    mma_ABt(C, A, B, C);
  }
  return C.tiles[0][0].data[0].x;
}

// Eight waves split K for one 16-channel output tile.  Each wave uses MFMA on
// a repeated activation row, then wave zero reduces the eight FP32 partials.
__global__ __launch_bounds__(64 * SPLIT_WAVES, 1)
void hk_bf16_matvec_splitk(bf16 *C_ptr, const bf16 *A_ptr, const bf16 *B_ptr, bf16 *unused) {
  constexpr int N = MATVEC_N, K = MATVEC_K;
  static_assert(N % 16 == 0 && K % (32 * SPLIT_WAVES) == 0);
  __shared__ st_bf<16, 32, st_16x32_s> As[SPLIT_WAVES];
  __shared__ st_bf<16, 32, st_16x32_s> Bs[SPLIT_WAVES];
  __shared__ float partial[SPLIT_WAVES][16];
  const int tid = threadIdx.x, wave = tid / 64, lane = tid & 63;
  const int out_base = blockIdx.x * 16;
  float result = 0.0f;
  switch (wave) {
    case 0: result = run_split<0>(A_ptr, B_ptr, out_base, As[0], Bs[0]); break;
    case 1: result = run_split<1>(A_ptr, B_ptr, out_base, As[1], Bs[1]); break;
    case 2: result = run_split<2>(A_ptr, B_ptr, out_base, As[2], Bs[2]); break;
    case 3: result = run_split<3>(A_ptr, B_ptr, out_base, As[3], Bs[3]); break;
    case 4: result = run_split<4>(A_ptr, B_ptr, out_base, As[4], Bs[4]); break;
    case 5: result = run_split<5>(A_ptr, B_ptr, out_base, As[5], Bs[5]); break;
    case 6: result = run_split<6>(A_ptr, B_ptr, out_base, As[6], Bs[6]); break;
    case 7: result = run_split<7>(A_ptr, B_ptr, out_base, As[7], Bs[7]); break;
  }
  if (lane < 16) partial[wave][lane] = result;
  asm volatile("s_waitcnt lgkmcnt(0)");
  __builtin_amdgcn_s_barrier();
  if (wave == 0 && lane < 16) {
    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < SPLIT_WAVES; i++) total += partial[i][lane];
    C_ptr[out_base + lane] = static_cast<bf16>(total);
  }
}
