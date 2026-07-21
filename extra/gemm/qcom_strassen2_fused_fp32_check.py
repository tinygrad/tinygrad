#!/usr/bin/env python3
"""Two-level Strassen full GEMM with arbitrary FP16 inputs and FP32 dot accumulation."""
import os

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import get_envelope, inject
from extra.gemm.qcom_strassen4_fused2_fp32_check import alloc, combine2_src, transform2_src


def main() -> None:
  n, seed = int(os.getenv("N", "1024")), int(os.getenv("SEED", "981"))
  if n not in (1024, 2048): raise ValueError("N must be 1024 or 2048")
  leaf = n//4
  wg, combine_wg = int(os.getenv("TRANSFORM_WG", "256")), int(os.getenv("COMBINE_WG", "128"))
  transform_vec, combine_vec = int(os.getenv("TRANSFORM_VEC", "2")), int(os.getenv("COMBINE_VEC", "4"))
  leaf_unroll = int(os.getenv("LEAF_UNROLL", "3"))
  rng = np.random.default_rng(seed)
  a_np = (rng.standard_normal((n, n), dtype=np.float32)*np.float32(1/32)).astype(np.float16)
  b_np = (rng.standard_normal((n, n), dtype=np.float32)*np.float32(1/32)).astype(np.float16)
  a, b = alloc(n*n, dtypes.half), alloc(n*n, dtypes.half)
  a.copyin(memoryview(a_np).cast("B")); b.copyin(memoryview(b_np).cast("B"))
  pa, pb, pm, c = (alloc(49*leaf*leaf, dtypes.half), alloc(49*leaf*leaf, dtypes.half),
                   alloc(49*leaf*leaf, dtypes.half), alloc(n*n, dtypes.float))
  dev = Device["QCOM"]
  transforms = {side: dev.runtime("transform", dev.compiler.compile(
    transform2_src(n, side, True, True, wg, transform_vec, False, False, True)), buf_dtypes=[
      ((0, dtypes.half, (49*leaf*leaf,)),), ((0, dtypes.half, (n*n,)),)]) for side in "AB"}
  combine = dev.runtime("combine", dev.compiler.compile(
    combine2_src(n, combine_wg, True, False, combine_vec, False)), buf_dtypes=[
      ((0, dtypes.float, (n*n,)),), ((0, dtypes.half, (49*leaf*leaf,)),)])

  q.M = q.N = q.K = leaf; q.K4 = leaf//4
  env, io, sz, ro = get_envelope(dev, q.make_direct_image_donor_src(2, 64))
  shader, hregs, fregs, loop_instrs = q.build_4x8_fp32_rotate_shader(
    dev, 64, k_count=leaf//4, batch_stride=leaf, batch_from_row=True, k_unroll=leaf_unroll)
  lib = inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs, mergedregs=False)
  gemms = {}
  def gemm(batch: int):
    if batch not in gemms:
      gemms[batch] = dev.runtime("gemm_h", lib, buf_dtypes=[
        ((0, dtypes.half, (batch*leaf, leaf//4, 4)),), ((0, dtypes.half, (batch*leaf, leaf//4, 4)),),
        ((1, dtypes.half, (batch*leaf, leaf//4, 4)),)])
    return gemms[batch]
  def grid(groups: int) -> tuple[int, int, int]:
    gx = min(groups, 1024)
    while groups % gx: gx -= 1
    return gx, groups//gx, 1

  transform_groups = leaf*leaf//transform_vec//wg
  combine_groups = leaf*leaf//combine_vec//combine_wg
  times = []
  times.append(transforms["A"](pa._buf, a._buf, global_size=grid(transform_groups), local_size=(wg, 1, 1), wait=True))
  times.append(transforms["B"](pb._buf, b._buf, global_size=grid(transform_groups), local_size=(wg, 1, 1), wait=True))
  input_bytes, output_bytes = leaf*leaf*2, leaf*leaf*2
  max_batch = 8192//leaf
  for first in range(0, 49, max_batch):
    batch = min(max_batch, 49-first)
    times.append(gemm(batch)(pm._buf.offset(first*output_bytes, batch*output_bytes),
      pa._buf.offset(first*input_bytes, batch*input_bytes), pb._buf.offset(first*input_bytes, batch*input_bytes),
      global_size=(max(1, leaf//256), batch*leaf//8, 1), local_size=(64, 1, 1), wait=True))
  times.append(combine(c._buf, pm._buf, global_size=grid(combine_groups), local_size=(combine_wg, 1, 1), wait=True))
  elapsed = sum(x for x in times if x is not None)
  got = np.empty((n, n), np.float32); c.copyout(memoryview(got).cast("B"))
  expected = a_np.astype(np.float32) @ b_np.astype(np.float32)
  delta = np.abs(got-expected); bad = ~np.isclose(got, expected, rtol=1e-3, atol=8e-3)
  print(f"shape={n}x{n}x{n} algorithm=strassen2 inputs=fp16 accumulate=fp32 elapsed_ms={elapsed*1e3:.3f} "
        f"gflops={2*n**3/elapsed/1e9:.1f} transform_ms={(times[0]+times[1])*1e3:.3f} "
        f"gemm_ms={sum(times[2:-1])*1e3:.3f} combine_ms={times[-1]*1e3:.3f} outputs={n*n} "
        f"bad_count={int(bad.sum())} max_abs={float(delta.max()):.9g} mean_abs={float(delta.mean()):.9g} "
        f"allclose={not bool(bad.any())} transform_wg={wg} combine_wg={combine_wg} transform_vec={transform_vec} "
        f"combine_vec={combine_vec} leaf_unroll={leaf_unroll} loop_instrs={loop_instrs}")
  if bad.any(): raise SystemExit(1)


if __name__ == "__main__": main()
