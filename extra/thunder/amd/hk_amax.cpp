// each workgroup scans multiple 16x32 tiles and computes max locally
// then every workgroup broadcasts their max through an atomic
#include "kittens.cuh"
using namespace kittens;

constexpr unsigned int TILE_R = 16, TILE_C = 32;
constexpr unsigned int ELEMS_PER_TILE = TILE_R * TILE_C;
constexpr unsigned int GRID = PARAM_GRID;

using ST = st_bf<TILE_R, TILE_C, st_16x32_s>;
using RT = rt_bf<TILE_R, TILE_C, row_l, rt_16x32_s>;
using G  = group<1>;

__device__ static inline void atomic_max_nonneg(float *dst, float v) {
  atomicMax(reinterpret_cast<unsigned int *>(dst), __float_as_uint(v));
}

extern "C" __global__ void custom_quantize_fp8_amax(float *amax_out, const bf16 *x) {
  constexpr unsigned int N = PARAM_N;
  constexpr unsigned int NUM_TILES = N / ELEMS_PER_TILE;

  gl<bf16, 1, 1, -1, -1> X{const_cast<bf16*>(x), nullptr, nullptr, (size_t)(N / TILE_C), (size_t)TILE_C};

  __shared__ ST smem;
  RT reg;
  typename RT::col_vec row_max_vec;

  float block_max = 0.0f;

  for (unsigned int tile_idx = blockIdx.x; tile_idx < NUM_TILES; tile_idx += GRID) {
    G::load(smem, X, {0, 0, (int)tile_idx, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();

    load(reg, smem);
    abs(reg, reg);
    row_max(row_max_vec, reg);

    bf16 tile_max_bf = bf16(0.0f);
    max(tile_max_bf, row_max_vec);
    block_max = fmaxf(block_max, (float)tile_max_bf);
  }

  if (laneid() == 0) atomic_max_nonneg(amax_out, block_max);
}
