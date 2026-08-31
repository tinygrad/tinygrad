#!/usr/bin/env python3
"""Correctness and DEBUG=2 performance verification for MI350X MXFP4 target GEMMs."""

import gc, math


M, K = 16384, 4096
TARGETS = (28672, 14336, 4096, 6144)
WARMUPS, ITERATIONS = 1, 7
MAX_REL_ERROR = 0.2


def main() -> None:
  from tinygrad import Context, Device, Tensor, dtypes
  from tinygrad.helpers import DEBUG
  from extra.gemm.cdna_asm_gemm import _mxfp4_gemm_quantized
  from extra.llama_kernels.quantize_mxfp4 import quantize_mxfp4

  if not Device.DEFAULT.startswith("AMD"):
    raise RuntimeError(f"requires DEV=AMD, got {Device.DEFAULT}")
  if DEBUG.value < 2:
    raise RuntimeError("run with DEBUG=2 so mxfp4_gemm_* PFLOPS are printed")

  print(f"MXFP4 target verification: M={M} K={K}, warmups={WARMUPS}, iterations={ITERATIONS}")
  for target_index, n in enumerate(TARGETS):
    Tensor.manual_seed(100 + target_index)
    with Context(DEBUG=0):
      a = Tensor.randn(M, K, dtype=dtypes.float32).cast(dtypes.bfloat16).realize()
      b = Tensor.randn(n, K, dtype=dtypes.float32).cast(dtypes.bfloat16).realize()
      a_q, scale_a, _, _ = quantize_mxfp4(a, shuffle_col=True)
      b_q, scale_b, _, _ = quantize_mxfp4(b, shuffle_row=True, shuffle_col=True)
      Tensor.realize(a_q, scale_a, b_q, scale_b)

    print(f"SHAPE M={M} N={n} K={K}")
    for iteration in range(WARMUPS + ITERATIONS):
      out = _mxfp4_gemm_quantized(a_q, b_q, scale_a, scale_b).realize()
      phase = "warmup" if iteration < WARMUPS else f"iteration {iteration - WARMUPS + 1}"
      print(f"  {phase}: GEMM complete")

    with Context(DEBUG=0):
      reference = (a @ b.T).realize()
      relative_error = ((out.float() - reference.float()).square().sum() / reference.float().square().sum()).sqrt().item()
    if not math.isfinite(relative_error) or relative_error >= MAX_REL_ERROR:
      raise AssertionError(f"N={n} relative error {relative_error:.6f} >= {MAX_REL_ERROR}")
    print(f"PASS M={M} N={n} K={K}: full_matrix_relative_error={relative_error:.6f}")
    del out, reference, a_q, scale_a, b_q, scale_b, a, b
    gc.collect()

  print("ALL TARGET SHAPES PASS")


if __name__ == "__main__": main()
