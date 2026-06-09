/**
 * @file
 * @brief MXFP8 block scale loading and packing utilities.
 *
 * Provides functions for staging E8M0 block scales in LDS and packing them
 * into fp8e8m0_4 registers for use with scaled MFMA instructions.
 */

#pragma once

#include "../../../../common/common.cuh"

namespace kittens {

/**
 * @brief Load iteration-major packed E8M0 scales from global memory into LDS.
 *
 * First 256 threads each load one uint32 (4 packed E8M0 bytes) for A and B.
 * A scales are placed at smem[0..1023], B scales at smem[1024..2047].
 *
 * @param smem_scales LDS buffer, must be >= 2048 bytes.
 * @param scale_A_iter Iteration-major A scales: [k_iter * M + row] as uint32.
 * @param scale_B_iter Iteration-major B scales: [k_iter * N + row] as uint32.
 * @param block_m Starting row offset for A within the current block.
 * @param block_n Starting row offset for B within the current block.
 * @param k_iter Current K iteration index.
 * @param M_dim M dimension of the matrix.
 * @param N_dim N dimension of the matrix.
 */
__device__ __forceinline__ void load_scales_to_lds(
    uint8_t *smem_scales,
    const uint32_t *__restrict__ scale_A_iter,
    const uint32_t *__restrict__ scale_B_iter,
    int block_m, int block_n, int k_iter, int M_dim, int N_dim) {
    int tid = threadIdx.x;
    if (tid < 256) {
        uint32_t sa = scale_A_iter[k_iter * M_dim + block_m + tid];
        uint32_t sb = scale_B_iter[k_iter * N_dim + block_n + tid];
        *(uint32_t *)&smem_scales[tid * 4] = sa;
        *(uint32_t *)&smem_scales[1024 + tid * 4] = sb;
    }
}

/**
 * @brief Pack 4 E8M0 scale bytes from LDS into one fp8e8m0_4 register.
 *
 * Each lane (r16 = laneid%16, k_sub = laneid/16) loads 4 dwords from
 * consecutive 16-row groups, then uses v_perm_b32 to extract byte k_sub
 * from each, producing the packed scale register for scaled MFMA.
 *
 * @param smem_scales LDS pointer to scale region.
 * @param lds_base Byte offset within smem_scales (0 for A, 1024 for B).
 * @param row_offset Starting row within the scale region (warp's tile offset).
 * @return fp8e8m0_4 with 4 scale bytes packed for MFMA opsel.
 */
__device__ __forceinline__ fp8e8m0_4 pack_scales(
    const uint8_t *smem_scales, int lds_base, int row_offset) {
    int lid   = laneid();
    int r16   = lid % 16;
    int k_sub = lid / 16;

    const uint32_t *s32 = (const uint32_t *)(smem_scales + lds_base);
    uint32_t w0 = s32[row_offset + 0 * 16 + r16];
    uint32_t w1 = s32[row_offset + 1 * 16 + r16];
    uint32_t w2 = s32[row_offset + 2 * 16 + r16];
    uint32_t w3 = s32[row_offset + 3 * 16 + r16];

    uint32_t sel = 0x0C0C0000u | (k_sub << 8) | (4u + k_sub);
    uint32_t lo  = __builtin_amdgcn_perm(w0, w1, sel);
    uint32_t hi  = __builtin_amdgcn_perm(w2, w3, sel);

    return (fp8e8m0_4)(lo | (hi << 16));
}
} // namespace kittens
