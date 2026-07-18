#!/usr/bin/env python3
"""Dependency-free lane-mapping probe for the thread-major 8x8 shader."""
import ctypes, os, random, struct

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_8x4_gemm as q8
from extra.gemm.ir3asm import get_envelope, inject


def half_bytes(values): return bytearray(struct.pack(f"<{len(values)}e", *values))


def main():
  m, n, k = int(os.getenv("M", "32")), int(os.getenv("N", "256")), int(os.getenv("K", "192"))
  pattern = os.getenv("PATTERN", "row")
  int8_b = bool(int(os.getenv("INT8_B", "0")))
  a = [0.0] * (m*k)
  b = [0.0] * (k*n)
  if pattern == "row":
    for row in range(m): a[row*k] = row+1
    for col in range(n): b[col] = 1
  elif pattern == "col":
    for row in range(m): a[row*k] = 1
    for col in range(n): b[col] = col % 251 + 1
  elif pattern == "random":
    rng = random.Random(int(os.getenv("SEED", "0")))
    a = [rng.uniform(-0.05, 0.05) for _ in a]
    b = [rng.uniform(-0.05, 0.05) for _ in b]
  else: raise ValueError(pattern)
  # The oracle must use the exact FP16 values consumed by the images.
  a = list(struct.unpack(f"<{len(a)}e", half_bytes(a)))
  if int8_b:
    bq = [max(-127, min(127, round(x*127))) for x in b]
    b = [x/127.0 for x in bq]
    b_bytes = bytearray((x & 0xff) for x in bq)
  else:
    b = list(struct.unpack(f"<{len(b)}e", half_bytes(b)))
    b_bytes = half_bytes(b)

  q8.M, q8.N, q8.K, q8.K4 = m, n, k, k//4
  dev = Device["QCOM"]
  compiler = bool(int(os.getenv("COMPILER", "0")))
  tight_store = bool(int(os.getenv("TIGHT_STORE", "0")))
  mode = os.getenv("MODE", "")
  if compiler:
    lib, _, _, _ = get_envelope(dev, q8.make_donor_src8(2, 128))
  elif mode:
    env, io, sz, ro = get_envelope(dev, q8.make_donor_src8(4, 128))
    if mode == "pipeline": shader, hregs, fregs, _, _ = q8.build_8x8_pipelined_shader(dev, 128, 4, 4, thread_store_gx=n//256)
    elif mode == "pipeline4": shader, hregs, fregs, _, _ = q8.build_8x8_pipeline4_shader(dev, 128, 4, 4)
    elif mode == "batch2": shader, hregs, fregs, _, _ = q8.build_8x8_batch2_shader(dev, 128, 4, 4)
    else: raise ValueError(mode)
    assert len(shader) <= sz
    lib = inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs)
  elif int(os.getenv("BASE", "0")):
    env, io, sz, ro = get_envelope(dev, q8.make_donor_src8(4, 128))
    shader, hregs, fregs, _ = q8.build_8x8_split_a_shader(dev, 128,
      a_coord_delay=int(os.getenv("ADELAY", "3")), b_coord_delay=int(os.getenv("BDELAY", "3")),
      pre_mad_nops=int(os.getenv("PMAD", "-1")), grouped_b=bool(int(os.getenv("GROUPED_B", "0"))),
      grouped_b_cols=bool(int(os.getenv("GROUPED_COLS", "0"))), thread_store_gx=0 if tight_store else 1,
      add256_store_mode="tight" if tight_store else "donor")
    lib = inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs)
  else:
    env, io, sz, ro = get_envelope(dev, q8.make_donor_src8(4, 128))
    shader, hregs, fregs, _ = q8.build_8x8_split_a_unroll_shader(dev, 128,
      k_unroll=int(os.getenv("KUNROLL", "8")), b_coord_delay=int(os.getenv("BDELAY", "0")),
      fast_coords=bool(int(os.getenv("FAST", "1"))), prefetch_next_b=bool(int(os.getenv("PREFETCH", "0"))),
      thread_store_gx=0 if tight_store else 1, add256_store_mode="tight" if tight_store else "donor",
      post_sequence=bool(int(os.getenv("POST_SEQUENCE", "0"))), a_coord_delay=int(os.getenv("ADELAY", "4")),
      unroll_gap=int(os.getenv("GAP", "0")), relaxed_sync=bool(int(os.getenv("RELAXED_SYNC", "0"))),
      sync_mask=int(os.getenv("SYNC_MASK", "7"), 0), sync_wait=int(os.getenv("SYNC_WAIT", "0")))
    lib = inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs)
  ab = Buffer("QCOM", len(a), dtypes.half).allocate()
  bb = Buffer("QCOM", len(b), dtypes.int8 if int8_b else dtypes.half).allocate()
  cb = Buffer("QCOM", m*n, dtypes.half).allocate()
  for buf, raw in ((ab, half_bytes(a)), (bb, b_bytes), (cb, bytearray(m*n*2))):
    src = (ctypes.c_ubyte * len(raw)).from_buffer(raw)
    ctypes.memmove(int(buf._buf.va_addr), ctypes.addressof(src), len(raw))
  specs = [((0, dtypes.half, (m, k//4, 4)),), ((0, dtypes.int8 if int8_b else dtypes.half, (k, n//4, 4)),),
           ((0, dtypes.half, None),)]
  prg = dev.runtime("gemm_h", lib, buf_dtypes=specs)
  times = [prg(ab._buf, bb._buf, cb._buf, global_size=(n//256, m//32, 1), local_size=(128, 1, 1), wait=True)*1e3 for _ in range(5)]
  print("elapsed_ms=", min(times))
  out = bytearray(m*n*2)
  ctypes.memmove(ctypes.addressof((ctypes.c_ubyte * len(out)).from_buffer(out)), int(cb._buf.va_addr), len(out))
  raw = struct.unpack(f"<{m*n}e", out)
  if int(os.getenv("DUMP_RAW", "0")):
    for row in range(min(m, 16)): print("raw", row, list(raw[row*n:row*n+min(n, 64)]))
    return
  if pattern == "random":
    worst = total = 0.0
    worst_at = None
    for row in range(m):
      tm, rr = row//8, row%8
      for col in range(n):
        tid, cc, lane = (col//4)%32, col//128, col%4
        got = raw[row*n+col] if compiler or tight_store or mode in ("pipeline4", "batch2") else raw[(tm*32+tid)*64 + rr*8 + cc*4 + lane]
        expected = sum(a[row*k+kk] * b[kk*n+col] for kk in range(k))
        delta = abs(got-expected)
        if delta > worst: worst, worst_at = delta, (row, col, got, expected)
        total += delta
    print("max_abs=", worst, "mean_abs=", total/(m*n), "worst_at=", worst_at)
    if worst > 0.02: raise SystemExit(1)
    return
  for lid in range(min(128, m*n//64)):
    vals = [raw[lid*64+row*8] for row in range(8)]
    print(lid, vals)


if __name__ == "__main__": main()
