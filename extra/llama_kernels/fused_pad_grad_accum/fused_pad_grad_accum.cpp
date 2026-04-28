// Fused custom kernel: grad_buf += cat(*chunks, dim=0) in one HBM pass.
//
// Template source — chunk parameter list and switch dispatch are filled by codegen
// in cast_amax.py:_build_fused_pad_grad_accum_src to support arbitrary N.
//
// Defines required at compile time:
//   CHUNK_SIZE       elements per chunk (must be multiple of THREADS_PER_WG * ELEMS_PER_THREAD)
//   THREADS_PER_WG
//   ELEMS_PER_THREAD (8 = one uint4 per thread = 16-byte vectorized load)
//
// Layout: one block-per-(slice-of-chunk) — blockIdx.x / BLOCKS_PER_CHUNK selects the chunk.
// All threads in a block read the same chunk → switch is uniform → no warp divergence.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#ifndef THREADS_PER_WG
#define THREADS_PER_WG 256
#endif
#ifndef ELEMS_PER_THREAD
#define ELEMS_PER_THREAD 8
#endif

#define ELEMS_PER_BLOCK (THREADS_PER_WG * ELEMS_PER_THREAD)
#define BLOCKS_PER_CHUNK (CHUNK_SIZE / ELEMS_PER_BLOCK)

extern "C" __attribute__((global))
__attribute__((amdgpu_flat_work_group_size(1, THREADS_PER_WG)))
void fused_pad_grad_accum(
    __hip_bfloat16* __restrict__ grad_buf
    __FUSED_PAD_GRAD_ACCUM_PARAMS
) {
  const int bid = blockIdx.x;
  const int chunk_idx = bid / BLOCKS_PER_CHUNK;
  const int block_in_chunk = bid - chunk_idx * BLOCKS_PER_CHUNK;
  const int tid = threadIdx.x;

  const __hip_bfloat16* chunk_ptr;
  switch (chunk_idx) {
    __FUSED_PAD_GRAD_ACCUM_DISPATCH
    default: chunk_ptr = (const __hip_bfloat16*)0; break;  // unreachable
  }

  // int64 for global_offset: at 32 chunks × 117M elements = 3.6B, int32 overflows → MEMVIOL.
  const int local_offset = block_in_chunk * ELEMS_PER_BLOCK + tid * ELEMS_PER_THREAD;
  const long long global_offset = (long long)chunk_idx * (long long)CHUNK_SIZE + (long long)local_offset;

  // Vectorized 16-byte load (uint4 = 8 bf16). Requires CHUNK_SIZE % 8 == 0 and 16-byte alignment.
  const uint4 chunk_v = *reinterpret_cast<const uint4*>(&chunk_ptr[local_offset]);
  const uint4 grad_v  = *reinterpret_cast<const uint4*>(&grad_buf[global_offset]);
  uint4 out_v;

  const __hip_bfloat16* chunk_bf = reinterpret_cast<const __hip_bfloat16*>(&chunk_v);
  const __hip_bfloat16* grad_bf  = reinterpret_cast<const __hip_bfloat16*>(&grad_v);
  __hip_bfloat16*       out_bf   = reinterpret_cast<__hip_bfloat16*>(&out_v);

  #pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; i++) {
    out_bf[i] = (__hip_bfloat16)((float)grad_bf[i] + (float)chunk_bf[i]);
  }

  *reinterpret_cast<uint4*>(&grad_buf[global_offset]) = out_v;
}
