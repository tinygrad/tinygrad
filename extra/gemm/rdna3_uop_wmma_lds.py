"""RDNA3 half GEMM UOp reference: tid-partitioned WG LDS fill (≠ WMMA-swizzle STAGE bounce).

Hand kernel (`rdna3_asm_wmma_gemm.py`) and float UOp (`amd_uop_matmul.py`) share A/B through
LDS by writing with flat `tid` and reading with warp/lane/upcast. Codegen `TC_LDS_AB` stages
already-swizzled WMMA fragments with the same axes for write and read → bounce (LOAD≈LLOAD).

This file is the explicit UOp prototype for the real coop pattern before teaching STAGE/codegen.

  # MOCK gate (macOS): coop fill must show far fewer GLOBAL_LOAD than bounce TC_LDS_AB
  PYTHONPATH=. python extra/gemm/rdna3_uop_wmma_lds.py --count

  # COMPUTE=1 compile gate (MOCK)
  PYTHONPATH=. COMPUTE=1 python extra/gemm/rdna3_uop_wmma_lds.py --count

  # hardware (optional, after count gate + COMPUTE=1 correctness)
  # Hand WMMA uses float.vec(8); SPEC bans vec dtypes in the tensor graph — SPEC=0 required.
  DEV=AMD:AMD COMPUTE=1 VERIFY=1 PYTHONPATH=. python extra/gemm/rdna3_uop_wmma_lds.py
"""
from __future__ import annotations
import argparse, os
os.environ.setdefault("AMD_LLVM", "0")

from tinygrad import Tensor, Context, GlobalCounters
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import DEBUG, Target, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

# Match hand kernel tile / WG.
WARP_SIZE = 32
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 16
THREADS = 128
TC_M, TC_N, TC_K = 16, 16, 16
TILES_M = BLOCK_M // TC_M // 2  # 4 tiles per wave_m half
TILES_N = BLOCK_N // TC_N // 2
ELEM_PER_THREAD = (BLOCK_M * BLOCK_K) // THREADS  # 16 halfs = 32B = hand b128
assert ELEM_PER_THREAD == 16 and (BLOCK_N * BLOCK_K) // THREADS == 16

N = getenv("N", 256)
M, K = getenv("M", N), getenv("K", N)
assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0

# RDNA3 WMMA: A/B half.vec(16) via STACK; C float.vec(8). threads=32.
WMMA_ARG = ("WMMA_16_16_16_half_float", (16, 16, 16), dtypes.half, dtypes.float, "AMD", 32, ((), (), ()), ())

def tid_partitioned_fill(As:UOp, Bs:UOp, A:UOp, B:UOp, tid:UOp) -> UOp:
  """GLOBAL → LOCAL with flat tid. Writers ≠ WMMA readers."""
  A_view = A.reshape(THREADS, ELEM_PER_THREAD)
  B_view = B.reshape(THREADS, ELEM_PER_THREAD)
  As_view = As.reshape(THREADS, ELEM_PER_THREAD)
  Bs_view = Bs.reshape(THREADS, ELEM_PER_THREAD)
  inner = UOp.range(ELEM_PER_THREAD, 101, AxisType.UPCAST)
  As_store = As_view[tid, inner].store(A_view[tid, inner])
  Bs_store = Bs_view[tid, inner].store(B_view[tid, inner])
  return UOp.barrier(UOp.group(As_store, Bs_store).end(inner))

def custom_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  gx, gy = UOp.special(M // BLOCK_M, "gidx0"), UOp.special(N // BLOCK_N, "gidx1")
  k_tile = UOp.range(K // BLOCK_K, 0, AxisType.REDUCE)

  A = A.reshape((M // BLOCK_M, BLOCK_M, K // BLOCK_K, BLOCK_K))[gx, :, k_tile, :]
  B = B.reshape((K // BLOCK_K, BLOCK_K, N // BLOCK_N, BLOCK_N))[k_tile, :, gy, :]

  tid = UOp.special(THREADS, "lidx0")
  As = UOp.placeholder((BLOCK_M, BLOCK_K), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  Bs = UOp.placeholder((BLOCK_K, BLOCK_N), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)
  barrier = tid_partitioned_fill(As, Bs, A, B, tid)

  if getenv("COMPUTE", 0):
    As, Bs = As.after(barrier), Bs.after(barrier)
    # Scalar float acc[TILES_M][TILES_N][8] — avoids SPEC reject on float.vec(8) REG BUFFER.
    acc = UOp.placeholder((TILES_M, TILES_N, 8), dtypes.float, 0, AddrSpace.REG)
    acc = acc[zi:=UOp.range(TILES_M * TILES_N * 8, 50)].set(0.0, end=zi)

    warp, lane = tid // WARP_SIZE, tid % WARP_SIZE
    wave_m, wave_n = warp // 2, warp % 2
    lane16 = lane % 16

    # Explicit 4×4 unroll (hand kernel). UPCAST of tm×tn fights WMMA broadcast shapes.
    As_r = As.reshape(2, TILES_M, TC_M, TC_K)
    Bs_r = Bs.reshape(TC_K, 2, TILES_N, TC_N)
    stores = []
    for tm in range(TILES_M):
      for tn in range(TILES_N):
        la = UOp.range(TC_K, 220 + tm*10 + tn, AxisType.UPCAST)
        lb = UOp.range(TC_K, 320 + tm*10 + tn, AxisType.UPCAST)
        A_in = As_r[wave_m, tm, lane16, la].contract(la)
        B_in = Bs_r[lb, wave_n, tn, lane16].contract(lb)
        acc_load = UOp.vectorize(*[acc.after(k_tile)[tm, tn, i] for i in range(8)])
        out = UOp(Ops.WMMA, dtypes.float.vec(8), (A_in, B_in, acc_load), arg=WMMA_ARG)
        stores.extend([acc[tm, tn, i].store(out.index(i)) for i in range(8)])
    acc = acc.after(UOp.group(*stores).barrier().end(k_tile))

    C = C.reshape((M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N))
    ep = []
    for tm in range(TILES_M):
      for tn in range(TILES_N):
        for i in range(8):
          ep.append(C[gx, wave_m*64 + tm*TC_M + (lane//16) + i*2, gy, wave_n*64 + tn*TC_N + lane16].store(acc[tm, tn, i]))
    sink = UOp.group(*ep)
  else:
    C = C.reshape((M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N))
    peer = (tid + 1) % THREADS
    Asf, Bsf = As.reshape(THREADS, ELEM_PER_THREAD), Bs.reshape(THREADS, ELEM_PER_THREAD)
    sink = C.after(barrier.end(k_tile))[gx, 0, gy, 0].store(
      Asf[peer, 0].cast(dtypes.float) + Bsf[peer, 0].cast(dtypes.float))

  return sink.sink(arg=KernelInfo(name="rdna3_uop_wmma_lds", opts_to_apply=())).simplify()

def _count_mem(prg) -> dict[str, int]:
  from tinygrad.renderer.isa.amd import AMDOps
  from tinygrad.uop import Ops as UOps
  lin = [u.arg for u in prg.src[1].src if u.op is UOps.INS]
  return {
    "LOAD": lin.count(AMDOps.LOAD), "LLOAD": lin.count(AMDOps.LLOAD),
    "LSTORE": lin.count(AMDOps.LSTORE), "BARRIER": lin.count(AMDOps.BARRIER),
    "WMMA": lin.count(AMDOps.WMMA), "SPILL": lin.count(AMDOps.SPILL),
  }

def count_gate():
  """Compile UOp kernel + bounce TC_LDS_AB; print mem op counts (MOCK-friendly)."""
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.renderer.isa.amd import AMDRenderer

  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  to_program_cache.clear()
  compute = getenv("COMPUTE", 0)

  C = UOp.placeholder((M, N), dtypes.float, 0)
  A = UOp.placeholder((M, K), dtypes.half, 1)
  B = UOp.placeholder((K, N), dtypes.half, 2)
  sink = custom_gemm(C, A, B)
  # Hand WMMA+STACK operands match mi350; SPEC tensor rules reject them (TC path expands first).
  with Context(SPEC=0):
    prg = to_program(sink, ren)
  coop = _count_mem(prg)
  print(f"coop tid-fill (COMPUTE={compute}) @ {M}x{N}x{K}: {coop} local={prg.arg.local_size}")

  old = os.environ.get("TC_LDS_AB")
  os.environ["TC_LDS_AB"] = "1"
  getenv.cache_clear()
  to_program_cache.clear()
  try:
    with Context(BEAM=0):
      ast = (Tensor.empty(M, K, dtype=dtypes.half, device="AMD") @
             Tensor.empty(K, N, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
      lds = to_program(ast.schedule_linear().src[-1].src[0], ren)
    lcnt = _count_mem(lds)
    print(f"TC_LDS_AB matmul @ {M}x{N}: {lcnt}")
  finally:
    if old is None: os.environ.pop("TC_LDS_AB", None)
    else: os.environ["TC_LDS_AB"] = old
    getenv.cache_clear()
    to_program_cache.clear()

  # Fill-only UOp: many LSTORE, few LLOAD (write tile, peek peer). COMPUTE: LLOAD>LSTORE.
  # Codegen TC_LDS_AB: real coop LOAD<<LLOAD and LLOAD>LSTORE.
  ok_uop = (coop["LLOAD"] > coop["LSTORE"]) if compute else (coop["LSTORE"] > coop["LLOAD"] and coop["LOAD"] > 0)
  ok_wmma = (not compute) or coop["WMMA"] >= 16
  ok_codegen = lcnt["LLOAD"] > lcnt["LSTORE"] and lcnt["LOAD"] < lcnt["LLOAD"] and lcnt["SPILL"] == 0
  if ok_uop and ok_wmma and ok_codegen:
    print(f"GATE OK: uop LOAD {coop['LOAD']} LLOAD {coop['LLOAD']} LSTORE {coop['LSTORE']}; "
          f"codegen LOAD {lcnt['LOAD']} LLOAD {lcnt['LLOAD']} > LSTORE {lcnt['LSTORE']}"
          + (f"; WMMA {coop['WMMA']}" if compute else ""))
  else:
    print(f"GATE PENDING: coop={coop} codegen={lcnt}")

def run_kernel():
  a = Tensor.randn(M, K, dtype=dtypes.half)
  b = Tensor.randn(K, N, dtype=dtypes.half)
  c = Tensor.empty(M, N, dtype=dtypes.float)
  with Context(DEBUG=0): Tensor.realize(a, b)
  GlobalCounters.reset()
  # SPEC bans vec dtypes; hand WMMA is float.vec(8) with STACK operands (same as mi350x_uop).
  ctx = dict(DEBUG=max(2, DEBUG.value))
  if getenv("COMPUTE", 0): ctx["SPEC"] = 0
  with Context(**ctx):
    tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0].realize()
  if GlobalCounters.time_sum_s:
    print(f"{(N*M*K*2 / GlobalCounters.time_sum_s)*1e-12:.2f} REAL TFLOPS")
  if getenv("VERIFY", 0) and getenv("COMPUTE", 0):
    with Context(DEBUG=0):
      ref = (a.float() @ b.float()).realize()
      err = (ref - tst).square().mean().item()
    print(f"mse {err}")
    if err > 1e-2: raise RuntimeError("matmul wrong")

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--count", action="store_true", help="MOCK compile gate: LOAD vs bounce")
  args = ap.parse_args()
  if args.count: count_gate()
  else: run_kernel()
