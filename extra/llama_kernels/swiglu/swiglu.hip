/*************************************************************************
 * This file was modified for tinygrad integration.
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Specialized BF16 form of TransformerEngine's vectorized SwiGLU kernels.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#ifndef N_ELEMS
#define N_ELEMS 234881024
#endif
#ifndef HIDDEN
#define HIDDEN 14336
#endif
#ifndef NUM_WG
#define NUM_WG 1024
#endif
#ifndef THREADS_PER_WG
#define THREADS_PER_WG 512
#endif

constexpr int VEC = 16;  // TransformerEngine uses 32-byte vectors.
static_assert(N_ELEMS % VEC == 0);
static_assert(HIDDEN % VEC == 0);

union bf16x8 {
  float4 packed;
  __hip_bfloat16 elem[8];
};

__device__ __forceinline__ float sigmoidf(const float x) {
  return __frcp_rn(1.0f + __expf(-x));
}

#ifndef SWIGLU_BACKWARD
extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
swiglu_fwd(__hip_bfloat16* __restrict__ output,
           const __hip_bfloat16* __restrict__ input) {
  const int gid = blockIdx.x * THREADS_PER_WG + threadIdx.x;
  constexpr int stride = NUM_WG * THREADS_PER_WG * VEC;

  for (int base = gid * VEC; base < N_ELEMS; base += stride) {
    const int row = base / HIDDEN;
    const int col = base % HIDDEN;
    const int act_off = row * 2 * HIDDEN + col;
    const int gate_off = act_off + HIDDEN;

    bf16x8 act0{*reinterpret_cast<const float4*>(&input[act_off])};
    bf16x8 act1{*reinterpret_cast<const float4*>(&input[act_off + 8])};
    bf16x8 gate0{*reinterpret_cast<const float4*>(&input[gate_off])};
    bf16x8 gate1{*reinterpret_cast<const float4*>(&input[gate_off + 8])};
    bf16x8 out0, out1;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const float x0 = static_cast<float>(act0.elem[i]);
      const float x1 = static_cast<float>(act1.elem[i]);
      out0.elem[i] = static_cast<__hip_bfloat16>((x0 * sigmoidf(x0)) * static_cast<float>(gate0.elem[i]));
      out1.elem[i] = static_cast<__hip_bfloat16>((x1 * sigmoidf(x1)) * static_cast<float>(gate1.elem[i]));
    }

    *reinterpret_cast<float4*>(&output[base]) = out0.packed;
    *reinterpret_cast<float4*>(&output[base + 8]) = out1.packed;
  }
}
#else
extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
swiglu_bwd(__hip_bfloat16* __restrict__ grad_input,
           const __hip_bfloat16* __restrict__ input,
           const __hip_bfloat16* __restrict__ grad_output) {
  const int gid = blockIdx.x * THREADS_PER_WG + threadIdx.x;
  constexpr int stride = NUM_WG * THREADS_PER_WG * VEC;

  for (int base = gid * VEC; base < N_ELEMS; base += stride) {
    const int row = base / HIDDEN;
    const int col = base % HIDDEN;
    const int act_off = row * 2 * HIDDEN + col;
    const int gate_off = act_off + HIDDEN;

    bf16x8 act0{*reinterpret_cast<const float4*>(&input[act_off])};
    bf16x8 act1{*reinterpret_cast<const float4*>(&input[act_off + 8])};
    bf16x8 gate0{*reinterpret_cast<const float4*>(&input[gate_off])};
    bf16x8 gate1{*reinterpret_cast<const float4*>(&input[gate_off + 8])};
    bf16x8 grad0{*reinterpret_cast<const float4*>(&grad_output[base])};
    bf16x8 grad1{*reinterpret_cast<const float4*>(&grad_output[base + 8])};
    bf16x8 dact0, dact1, dgate0, dgate1;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const float x0 = static_cast<float>(act0.elem[i]);
      const float x1 = static_cast<float>(act1.elem[i]);
      const float s0 = sigmoidf(x0);
      const float s1 = sigmoidf(x1);
      const float silu0 = x0 * s0;
      const float silu1 = x1 * s1;
      const float g0 = static_cast<float>(grad0.elem[i]);
      const float g1 = static_cast<float>(grad1.elem[i]);
      dact0.elem[i] = static_cast<__hip_bfloat16>(g0 * (s0 + silu0 * (1.0f - s0)) * static_cast<float>(gate0.elem[i]));
      dact1.elem[i] = static_cast<__hip_bfloat16>(g1 * (s1 + silu1 * (1.0f - s1)) * static_cast<float>(gate1.elem[i]));
      dgate0.elem[i] = static_cast<__hip_bfloat16>(g0 * silu0);
      dgate1.elem[i] = static_cast<__hip_bfloat16>(g1 * silu1);
    }

    *reinterpret_cast<float4*>(&grad_input[act_off]) = dact0.packed;
    *reinterpret_cast<float4*>(&grad_input[act_off + 8]) = dact1.packed;
    *reinterpret_cast<float4*>(&grad_input[gate_off]) = dgate0.packed;
    *reinterpret_cast<float4*>(&grad_input[gate_off + 8]) = dgate1.packed;
  }
}
#endif
