#!/usr/bin/env python3
"""Fully checked one-level Strassen FP16 GEMM for Adreno 630."""
import hashlib, os
import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_8x4_gemm as q8
from extra.gemm import qcom_intensity_gemm as q4
from extra.gemm.ir3asm import get_envelope, inject


PREP_SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void strassen_prep(__global const half *A, __global const half *B,
  __global half *A1, __global half *A2, __global half *A5, __global half *A6, __global half *A7,
  __global half *B1, __global half *B3, __global half *B4, __global half *B6, __global half *B7) {
  int i=get_global_id(0), r=i>>7, c=(i&127)<<2, o=r*512+c;
  half4 a11=vload4(0,A+r*1024+c), a12=vload4(0,A+r*1024+c+512);
  half4 a21=vload4(0,A+(r+512)*1024+c), a22=vload4(0,A+(r+512)*1024+c+512);
  half4 b11=vload4(0,B+r*1024+c), b12=vload4(0,B+r*1024+c+512);
  half4 b21=vload4(0,B+(r+512)*1024+c), b22=vload4(0,B+(r+512)*1024+c+512);
  vstore4(a11+a22,0,A1+o); vstore4(a21+a22,0,A2+o); vstore4(a11+a12,0,A5+o);
  vstore4(a21-a11,0,A6+o); vstore4(a12-a22,0,A7+o);
  vstore4(b11+b22,0,B1+o); vstore4(b12-b22,0,B3+o); vstore4(b21-b11,0,B4+o);
  vstore4(b11+b12,0,B6+o); vstore4(b21+b22,0,B7+o);
}
"""

POST_SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void strassen_post(__global const half *M1, __global const half *M2,
  __global const half *M3, __global const half *M4, __global const half *M5,
  __global const half *M6, __global const half *M7, __global half *C) {
  int i=get_global_id(0), r=i>>7, c=(i&127)<<2, o=r*512+c;
  half4 m1=vload4(0,M1+o),m2=vload4(0,M2+o),m3=vload4(0,M3+o),m4=vload4(0,M4+o);
  half4 m5=vload4(0,M5+o),m6=vload4(0,M6+o),m7=vload4(0,M7+o);
  vstore4(m1+m4-m5+m7,0,C+r*1024+c); vstore4(m3+m5,0,C+r*1024+c+512);
  vstore4(m2+m4,0,C+(r+512)*1024+c); vstore4(m1-m2+m3+m6,0,C+(r+512)*1024+c+512);
}
"""


def alloc_half(count: int) -> Buffer:
  return Buffer("QCOM", count, dtypes.half).allocate()


def vectorize(src: str, vec: int) -> str:
  if vec == 4: return src
  if vec not in (8, 16): raise ValueError("TRANSFORM_VEC must be 4, 8, or 16")
  log_cols = {8: 6, 16: 5}[vec]
  return (src.replace("i>>7", f"i>>{log_cols}").replace("(i&127)<<2", f"(i&{(512//vec)-1})<<{vec.bit_length()-1}")
          .replace("half4", f"half{vec}").replace("vload4", f"vload{vec}").replace("vstore4", f"vstore{vec}"))


def main() -> None:
  seed, runs = int(os.getenv("SEED", "307")), int(os.getenv("BENCH_RUNS", "10"))
  transform_vec = int(os.getenv("TRANSFORM_VEC", "4"))
  rng = np.random.default_rng(seed)
  a_np = (rng.standard_normal((1024, 1024))*0.05).astype(np.float16)
  b_np = (rng.standard_normal((1024, 1024))*0.05).astype(np.float16)
  dev = Device["QCOM"]
  a, b = alloc_half(a_np.size), alloc_half(b_np.size)
  a.copyin(memoryview(a_np).cast("B")); b.copyin(memoryview(b_np).cast("B"))
  tile_elems, tile_bytes = 512*512, 512*512*2
  pa, pb, pm = alloc_half(7*tile_elems), alloc_half(7*tile_elems), alloc_half(7*tile_elems)
  aa = [pa._buf.offset(i*tile_bytes, tile_bytes) for i in range(7)]
  bb = [pb._buf.offset(i*tile_bytes, tile_bytes) for i in range(7)]
  mm = [pm._buf.offset(i*tile_bytes, tile_bytes) for i in range(7)]
  c = alloc_half(1024*1024)

  gspec = ((0, dtypes.half, None),)
  prep = dev.runtime("strassen_prep", dev.compiler.compile(vectorize(PREP_SRC, transform_vec)), buf_dtypes=[gspec]*12)
  post = dev.runtime("strassen_post", dev.compiler.compile(vectorize(POST_SRC, transform_vec)), buf_dtypes=[gspec]*8)

  q8.M = q8.N = q8.K = 512; q8.K4 = 128
  env, io, sz, ro = get_envelope(dev, q4.make_direct_image_donor_src(4, 128))
  shader, hregs, fregs, loop_instrs = q8.build_8x8_persistent_shader(
    dev, 128, dynamic_a4_dual=True, image_store=True)
  lib = inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs, mergedregs=False)
  out_spec, std_spec, wide_spec = ((0, dtypes.half, (512, 128, 4)),), ((0, dtypes.half, (512, 128, 4)),), ((0, dtypes.half, (512, 256, 4)),)
  gemm_ss = dev.runtime("gemm_h", lib, buf_dtypes=[out_spec, std_spec, ((1, dtypes.half, (512, 128, 4)),)])
  gemm_ws = dev.runtime("gemm_h", lib, buf_dtypes=[out_spec, wide_spec, ((1, dtypes.half, (512, 128, 4)),)])
  gemm_sw = dev.runtime("gemm_h", lib, buf_dtypes=[out_spec, std_spec, ((1, dtypes.half, (512, 256, 4)),)])
  quadrant_bytes = (512*1024+512)*2
  a11, a22, b11, b22 = a._buf, a._buf.offset(quadrant_bytes), b._buf, b._buf.offset(quadrant_bytes)
  if int(os.getenv("PRINT_META", "0")):
    print("gemm_meta", fregs, hregs, len(shader), loop_instrs, hashlib.sha1(lib).hexdigest()[:8])

  def iteration() -> tuple[float, list[float], float]:
    transform_groups = 512*(512//transform_vec)//128
    tp = prep(a._buf, b._buf, aa[0], aa[1], aa[4], aa[5], aa[6], bb[0], bb[2], bb[3], bb[5], bb[6],
              global_size=(transform_groups, 1, 1), local_size=(128, 1, 1), wait=True)
    args = [(gemm_ss, aa[0], bb[0]), (gemm_sw, aa[1], b11), (gemm_ws, a11, bb[2]),
            (gemm_ws, a22, bb[3]), (gemm_sw, aa[4], b22), (gemm_ss, aa[5], bb[5]), (gemm_ss, aa[6], bb[6])]
    tg = [prg(mm[i], ax, bx, global_size=(2, 16, 1), local_size=(128, 1, 1), wait=True) for i, (prg, ax, bx) in enumerate(args)]
    to = post(*mm, c._buf, global_size=(transform_groups, 1, 1), local_size=(128, 1, 1), wait=True)
    return tp, tg, to

  for _ in range(2): iteration()
  measured = [iteration() for _ in range(runs)]
  totals = [tp+sum(tg)+to for tp, tg, to in measured]
  best_i = int(np.argmin(totals)); tp, tg, to = measured[best_i]; elapsed = totals[best_i]

  got = np.empty((1024, 1024), np.float16); c.copyout(memoryview(got).cast("B"))
  expected = a_np.astype(np.float32) @ b_np.astype(np.float32)
  delta = np.abs(got.astype(np.float32)-expected)
  correct = np.allclose(got, expected, rtol=2e-2, atol=2e-2)
  bad = ~np.isfinite(got) | (delta > .02)
  gflops = 2*1024**3/elapsed/1e9
  print(f"shape=1024x1024x1024 algorithm=strassen1 accumulate=fp16 elapsed_ms={elapsed*1e3:.3f} gflops={gflops:.1f} "
        f"prep_ms={tp*1e3:.3f} gemm_ms={sum(tg)*1e3:.3f} post_ms={to*1e3:.3f} "
        f"max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} allclose={correct}")
  print(f"bad_count={int(bad.sum())} gemm_parts_ms={[round(x*1e3, 3) for x in tg]}")
  if not correct: raise SystemExit(1)


if __name__ == "__main__": main()
