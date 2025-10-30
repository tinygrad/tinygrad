#include "kittens.cuh"
using namespace kittens;

constexpr int g_N = 8192;
constexpr int BLOCK_SIZE = 32;
constexpr int NUM_WORKERS = 4;
constexpr int PIPE_STAGES = 2;

using sub_tile = st_bf<BLOCK_SIZE,BLOCK_SIZE>;
using tile_gl =  gl<bf16, 1, 1, g_N, g_N>;

__launch_bounds__(NUM_WORKERS*WARP_THREADS, 1)
__global__ void kernel(bf16 *c_ptr, bf16 *a_ptr, bf16 *b_ptr) {
  tile_gl g_C{c_ptr, nullptr, nullptr, nullptr, nullptr};
  tile_gl g_A{a_ptr, nullptr, nullptr, nullptr, nullptr};
  tile_gl g_B{b_ptr, nullptr, nullptr, nullptr, nullptr};

  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int*)&__shm[0]);
  st_bf<BLOCK_SIZE,BLOCK_SIZE> (&As)[PIPE_STAGES] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, PIPE_STAGES>();
  st_bf<BLOCK_SIZE,BLOCK_SIZE> (&Bs)[PIPE_STAGES] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, PIPE_STAGES>();

  rt_bf<BLOCK_SIZE,BLOCK_SIZE> A_reg;
  rt_bf<BLOCK_SIZE,BLOCK_SIZE> B_reg;
  rt_bf<BLOCK_SIZE,BLOCK_SIZE, ducks::rt_layout::col> B_reg_col;
  rt_fl<BLOCK_SIZE,BLOCK_SIZE> C_accum;

  int col = blockIdx.x;
  int row = blockIdx.y;

  warp::zero(C_accum);
  int num_tiles = (g_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  warpgroup::load_async(As[0], g_A, {0, 0, row, 0});
  warpgroup::load_async(Bs[0], g_B, {0, 0, 0, col});
  load_async_wait();
  __syncthreads();

  for (int tile = 0; tile < num_tiles; ++tile) {
    int smem_idx = tile % PIPE_STAGES;
    int next_smem_idx = (tile + 1) % PIPE_STAGES;

    if (tile < num_tiles - 1) {
      warpgroup::load_async(As[next_smem_idx], g_A, {0, 0, row, tile + 1});
      warpgroup::load_async(Bs[next_smem_idx], g_B, {0, 0, tile + 1, col});
      load_async_wait<1>();
    } else load_async_wait();

    __syncthreads();

    warp::load(A_reg, As[smem_idx]);
    warp::load(B_reg, Bs[smem_idx]);
    warp::swap_layout(B_reg_col, B_reg);
    warp::mma_AB(C_accum, A_reg, B_reg_col, C_accum);

    __syncthreads();
  }
  warp::store(g_C, C_accum, {0, 0, row, col});
}
