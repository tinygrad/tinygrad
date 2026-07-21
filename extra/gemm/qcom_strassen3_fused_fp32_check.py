#!/usr/bin/env python3
"""Three-level Strassen full GEMM with arbitrary FP16 inputs and FP32 dot accumulation."""
import os

import numpy as np

from tinygrad import Device, dtypes
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import get_envelope, inject
from extra.gemm.qcom_strassen4_fused2_fp32_check import (
  alloc, combine1_batch_src, combine2_src, transform1_batch_src, transform2_src,
)


def main() -> None:
  n, seed = int(os.getenv("N", "1024")), int(os.getenv("SEED", "991"))
  if n != 1024: raise ValueError("this performance gate is intentionally fixed at N=1024")
  parent, leaf = n//4, n//8
  wg, combine_wg = int(os.getenv("TRANSFORM_WG", "256")), int(os.getenv("COMBINE_WG", "128"))
  transform_vec, combine_vec = int(os.getenv("TRANSFORM_VEC", "2")), int(os.getenv("COMBINE_VEC", "4"))
  leaf_unroll = int(os.getenv("LEAF_UNROLL", "3"))
  leaf_tile = os.getenv("LEAF_TILE", "4x8")
  rng = np.random.default_rng(seed)
  a_np = (rng.standard_normal((n, n), dtype=np.float32)*np.float32(1/32)).astype(np.float16)
  b_np = (rng.standard_normal((n, n), dtype=np.float32)*np.float32(1/32)).astype(np.float16)
  a, b = alloc(n*n, dtypes.half), alloc(n*n, dtypes.half)
  a.copyin(memoryview(a_np).cast("B")); b.copyin(memoryview(b_np).cast("B"))
  pcount = 49*parent*parent
  lcount = 343*leaf*leaf
  pa, pb, la, lb, lm, pm = (alloc(pcount, dtypes.half), alloc(pcount, dtypes.half),
                             alloc(lcount, dtypes.half), alloc(lcount, dtypes.half),
                             alloc(lcount, dtypes.half), alloc(pcount, dtypes.half))
  c = alloc(n*n, dtypes.float)
  dev = Device["QCOM"]
  transforms2 = {side: dev.runtime("transform2", dev.compiler.compile(
    transform2_src(n, side, True, True, wg, transform_vec, False, False, True)), buf_dtypes=[
      ((0, dtypes.half, (pcount,)),), ((0, dtypes.half, (n*n,)),)]) for side in "AB"}
  transforms1 = {side: dev.runtime("transform1", dev.compiler.compile(
    transform1_batch_src(parent, 49, side, wg, transform_vec)), buf_dtypes=[
      ((0, dtypes.half, (lcount,)),), ((0, dtypes.half, (pcount,)),)]) for side in "AB"}
  combine1 = dev.runtime("combine1", dev.compiler.compile(
    combine1_batch_src(parent, 49, combine_wg, combine_vec)), buf_dtypes=[
      ((0, dtypes.half, (pcount,)),), ((0, dtypes.half, (lcount,)),)])
  combine2 = dev.runtime("combine2", dev.compiler.compile(
    combine2_src(n, combine_wg, True, False, combine_vec, False)), buf_dtypes=[
      ((0, dtypes.float, (n*n,)),), ((0, dtypes.half, (pcount,)),)])

  q.M = q.N = q.K = leaf; q.K4 = leaf//4
  env, io, sz, ro = get_envelope(dev, q.make_direct_image_donor_src(2, 64))
  if leaf_tile == "4x8":
    shader, hregs, fregs, loop_instrs = q.build_4x8_fp32_rotate_shader(
      dev, 64, k_count=leaf//4, batch_stride=leaf, batch_from_row=True, k_unroll=leaf_unroll)
    rows_per_group = 8
  elif leaf_tile == "8x8":
    shader, hregs, fregs, loop_instrs = q.build_8x8_fp32_shader(dev, 64, batch_stride=leaf)
    rows_per_group = 16
  else:
    raise ValueError("LEAF_TILE must be 4x8 or 8x8")
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

  times = {"transform": 0.0, "gemm": 0.0, "combine": 0.0}
  t2groups = parent*parent//transform_vec//wg
  t1groups = 49*leaf*leaf//transform_vec//wg
  c1groups = 49*leaf*leaf//combine_vec//combine_wg
  c2groups = parent*parent//combine_vec//combine_wg
  for prg, out, inp in ((transforms2["A"], pa, a), (transforms2["B"], pb, b)):
    times["transform"] += prg(out._buf, inp._buf, global_size=grid(t2groups), local_size=(wg, 1, 1), wait=True)
  for prg, out, inp in ((transforms1["A"], la, pa), (transforms1["B"], lb, pb)):
    times["transform"] += prg(out._buf, inp._buf, global_size=grid(t1groups), local_size=(wg, 1, 1), wait=True)
  matrix_bytes = leaf*leaf*2
  max_batch = 8192//leaf
  for first in range(0, 343, max_batch):
    batch = min(max_batch, 343-first)
    span = batch*matrix_bytes
    times["gemm"] += gemm(batch)(lm._buf.offset(first*matrix_bytes, span), la._buf.offset(first*matrix_bytes, span),
      lb._buf.offset(first*matrix_bytes, span), global_size=(1, batch*leaf//rows_per_group, 1), local_size=(64, 1, 1), wait=True)
  times["combine"] += combine1(pm._buf, lm._buf, global_size=grid(c1groups), local_size=(combine_wg, 1, 1), wait=True)
  times["combine"] += combine2(c._buf, pm._buf, global_size=grid(c2groups), local_size=(combine_wg, 1, 1), wait=True)
  elapsed = sum(times.values())
  got = np.empty((n, n), np.float32); c.copyout(memoryview(got).cast("B"))
  expected = a_np.astype(np.float32) @ b_np.astype(np.float32)
  delta = np.abs(got-expected); bad = ~np.isclose(got, expected, rtol=1e-3, atol=8e-3)
  print(f"shape={n}x{n}x{n} algorithm=strassen3 inputs=fp16 accumulate=fp32 elapsed_ms={elapsed*1e3:.3f} "
        f"gflops={2*n**3/elapsed/1e9:.1f} transform_ms={times['transform']*1e3:.3f} "
        f"gemm_ms={times['gemm']*1e3:.3f} combine_ms={times['combine']*1e3:.3f} outputs={n*n} "
        f"bad_count={int(bad.sum())} max_abs={float(delta.max()):.9g} mean_abs={float(delta.mean()):.9g} "
        f"allclose={not bool(bad.any())} leaf_tile={leaf_tile} leaf_unroll={leaf_unroll} loop_instrs={loop_instrs}")
  if bad.any(): raise SystemExit(1)


if __name__ == "__main__": main()
