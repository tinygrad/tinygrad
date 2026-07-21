#!/usr/bin/env python3
"""Exact randomized oracle and benchmark for A630 packed UINT8 dp4acc GEMM."""
import os, random, struct

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import get_envelope, inject


def pack4(xs): return sum((int(x) & 0xff) << (8*i) for i, x in enumerate(xs))


def main():
  m, n, k = int(os.getenv("M", "16")), int(os.getenv("N", "128")), int(os.getenv("K", "192"))
  seed, check = int(os.getenv("SEED", "0")), bool(int(os.getenv("CHECK", "1")))
  rng = random.Random(seed)
  ones = bool(int(os.getenv("ONES", "0")))
  if check:
    signed_a = bool(int(os.getenv("SIGNED_A", "0")))
    val_range = int(os.getenv("VAL_RANGE", "8"))
    av = [1 if ones else rng.randrange(-val_range, val_range) if signed_a else rng.randrange(val_range) for _ in range(m*k)]
    bv = [1 if ones else rng.randrange(val_range) for _ in range(k*n)]
    ap = [pack4(av[row*k+ki*16+c*4:row*k+ki*16+c*4+4]) for row in range(m) for ki in range(k//16) for c in range(4)]
    bp = [pack4([bv[(ki*16+j*4+l)*n+col4*4+c] for l in range(4)])
          for ki in range(k//16) for j in range(4) for col4 in range(n//4) for c in range(4)]
  else:
    # Throughput-only runs do not need to spend O(MNK) time packing Python
    # integers. The shader executes the same instructions for zero words.
    av = bv = []
    ap, bp = [0] * (m*k//4), [0] * (k*n//4)
  combined = bool(int(os.getenv("COMBINED", "0")))
  if combined:
    width, bheight = max(k//16, n//4), k//4
    packed = [0] * ((bheight+m)*width*4)
    for y in range(bheight): packed[y*width*4:y*width*4+(n//4)*4] = bp[y*(n//4)*4:(y+1)*(n//4)*4]
    for row in range(m): packed[(bheight+row)*width*4:(bheight+row)*width*4+(k//16)*4] = ap[row*(k//16)*4:(row+1)*(k//16)*4]

  q.M, q.N, q.K, q.K4 = m, n, k, k//4
  dev = Device["QCOM"]
  env, io, sz, ro = get_envelope(dev, q.make_direct_donor_src_u32(1, 128))
  const_inputs, const_output = bool(int(os.getenv("CONST_INPUTS", "0"))), bool(int(os.getenv("CONST_OUTPUT", "0")))
  shader, hregs, fregs, _ = q.build_4x4_dp4_shader(dev, 128, k, constant_inputs=const_inputs, constant_output=const_output,
    constant_a=bool(int(os.getenv("CONST_A", "0"))), constant_b=bool(int(os.getenv("CONST_B", "0"))),
    combined_b_height=k//4 if combined else 0, mixed=bool(int(os.getenv("MIXED", "0"))),
    initial_acc=int(os.getenv("INITIAL_ACC", "0")), coord_delay=int(os.getenv("COORD_DELAY", "4")))
  assert len(shader) <= sz, (len(shader), sz)
  lib = inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs)
  ab, bb, cb = (Buffer("QCOM", size, dtype).allocate() for size, dtype in
                ((len(packed) if combined else len(ap), dtypes.uint32),
                 (len(packed) if combined else len(bp), dtypes.uint32), (m*n, dtypes.int32)))
  ab.copyin(memoryview(bytearray(struct.pack(f"<{ab.size}I", *(packed if combined else ap)))))
  bb.copyin(memoryview(bytearray(struct.pack(f"<{bb.size}I", *(packed if combined else bp)))))
  cb.copyin(memoryview(bytearray(m*n*4)))
  specs = ([((0, dtypes.uint32, (k//4+m, max(k//16, n//4), 4)),),
            ((1, dtypes.uint32, (k//4+m, max(k//16, n//4), 4)),)] if combined else
           [((0, dtypes.uint32, (m, k//16, 4)),), ((1, dtypes.uint32, (k//4, n//4, 4)),)]) + [((0, dtypes.int32, None),)]
  prg = dev.runtime("gemm_h", lib, buf_dtypes=specs)
  times = [prg(ab._buf, bb._buf, cb._buf, global_size=(n//128, m//16, 1), local_size=(128, 1, 1), wait=True) for _ in range(10)]
  print(f"elapsed_ms={min(times)*1e3:.4f} gops={2*m*n*k/min(times)/1e9:.1f}")
  if check:
    outb = bytearray(m*n*4)
    cb.copyout(memoryview(outb))
    got = struct.unpack(f"<{m*n}i", outb)
    worst = 0
    for row in range(m):
      for col in range(n):
        expected = sum(av[row*k+kk]*bv[kk*n+col] for kk in range(k)) + int(os.getenv("INITIAL_ACC", "0"))
        worst = max(worst, abs(got[row*n+col]-expected))
    print("first=", list(got[:16]))
    print("max_abs=", worst)
    if worst: raise SystemExit(1)


if __name__ == "__main__": main()
