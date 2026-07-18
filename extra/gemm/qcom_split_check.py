#!/usr/bin/env python3
"""Check a hand-assembled split-K GEMM against a NumPy partial matmul."""
import argparse

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import disasm, get_envelope, inject


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--m", type=int, default=128)
  parser.add_argument("--n", type=int, default=384)
  parser.add_argument("--ncols", type=int, default=0, help="128-column blocks computed by each workgroup")
  parser.add_argument("--k", type=int, default=1536)
  parser.add_argument("--k-start", type=int, default=0, help="first K/4 iteration")
  parser.add_argument("--k-count", type=int, default=48, help="number of K/4 iterations")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--scale", type=float, default=0.25)
  parser.add_argument("--impulse-k", type=int, default=-1,
                      help="zero both inputs except one K plane (useful for coordinate diagnostics)")
  parser.add_argument("--compiler", action="store_true", help="check the compiler donor over the full K range")
  parser.add_argument("--donor-store", action="store_true")
  parser.add_argument("--store-constant", action="store_true")
  parser.add_argument("--serial-b-cols", action="store_true")
  parser.add_argument("--disasm", action="store_true")
  args = parser.parse_args()
  assert args.m % 16 == 0 and args.n % 128 == 0 and args.n <= 512

  rng = np.random.default_rng(args.seed)
  a_np = (rng.standard_normal((args.m, args.k))*args.scale).astype(np.float16)
  b_np = (rng.standard_normal((args.k, args.n))*args.scale).astype(np.float16)
  if args.impulse_k >= 0:
    assert 0 <= args.impulse_k < args.k
    a_keep, b_keep = a_np[:, args.impulse_k].copy(), b_np[args.impulse_k].copy()
    a_np.fill(0)
    b_np.fill(0)
    a_np[:, args.impulse_k], b_np[args.impulse_k] = a_keep, b_keep
  dev = Device["QCOM"]
  q.M, q.N, q.K, q.K4 = args.m, 1024, args.k, args.k//4
  ncols = args.ncols or args.n//128
  assert args.n % (128*ncols) == 0
  if args.compiler:
    args.k_start, args.k_count = 0, args.k//4
    env, io, sz, _ = get_envelope(dev, q.make_direct_donor_src(ncols, 128))
    shader = bytes(env[io:io+sz])
    lib = bytes(env)
  else:
    # Keep enough instruction capacity for the explicitly initialized hand
    # shader even when it computes only one column block.
    env, io, sz, ro = get_envelope(dev, q.make_donor_src(max(ncols, 3), 128))
    unroll = 4 if args.k_count % 4 == 0 else 2 if args.k_count % 2 == 0 else 1
    shader, _ = q.build_4xn_shader(dev, 128, ncols=ncols, direct=True, compact_acc=True,
      alu_order="row_col_kk", k_unroll=unroll, first_sync_only=False, coord_delay=4, serial_b_cols=args.serial_b_cols,
      k_start=args.k_start, k_count=args.k_count, donor_store=args.donor_store,
      store_constant=args.store_constant)
    lib = inject(env, io, sz, ro, shader, fregs=10, hregs=20+4*ncols)
  if args.disasm:
    asm = disasm(shader)
    print(asm)

  a = Buffer("QCOM", a_np.size, dtypes.half).allocate()
  b = Buffer("QCOM", b_np.size, dtypes.half).allocate()
  c = Buffer("QCOM", args.m*1024, dtypes.half).allocate()
  a.copyin(memoryview(a_np).cast("B"))
  b.copyin(memoryview(b_np).cast("B"))
  c.copyin(memoryview(np.zeros(args.m*1024, dtype=np.float16)).cast("B"))
  buf_dtypes = [((0, dtypes.half, (args.m, args.k//4, 4)),),
                ((0, dtypes.half, (args.k, args.n//4, 4)),), ((0, dtypes.half, None),)]
  prg = dev.runtime("gemm_h", lib, buf_dtypes=buf_dtypes)
  elapsed = prg(a._buf, b._buf, c._buf, global_size=(args.n//(128*ncols), args.m//16, 1), local_size=(128, 1, 1), wait=True)
  got_flat = np.empty(args.m*1024, dtype=np.float16)
  c.copyout(memoryview(got_flat).cast("B"))
  got = got_flat.reshape(args.m, 1024)[:, :args.n].astype(np.float32)
  lo, hi = args.k_start*4, (args.k_start+args.k_count)*4
  expected = a_np[:, lo:hi].astype(np.float32) @ b_np[lo:hi].astype(np.float32)
  delta = np.abs(expected-got)
  worst = np.unravel_index(np.argmax(delta), delta.shape)
  def best_axis(left, right):
    left = left-left.mean(axis=1, keepdims=True)
    right = right-right.mean(axis=1, keepdims=True)
    corr = left@right.T/(np.linalg.norm(left, axis=1, keepdims=True)*np.linalg.norm(right, axis=1)[None, :]+1e-20)
    match = np.argmax(np.abs(corr), axis=1)
    return match, corr[np.arange(len(match)), match]
  row_match, row_corr = best_axis(got, expected)
  col_match, col_corr = best_axis(got.T, expected.T)
  print(f"elapsed_ms={elapsed*1e3:.3f} max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} "
        f"worst={worst} expected={expected[worst]!r} got={got[worst]!r}")
  print(f"rows match={row_match[:16].tolist()} corr={np.round(row_corr[:16], 3).tolist()} median={np.median(np.abs(row_corr)):.3f}")
  print(f"cols match={col_match[:16].tolist()} corr={np.round(col_corr[:16], 3).tolist()} median={np.median(np.abs(col_corr)):.3f}")
  print(f"expected[0,:8]={expected[0, :8].tolist()} got[0,:8]={got[0, :8].tolist()} "
        f"within4_deltas={[float(np.max(np.abs(got[:, 0]-got[:, i]))) for i in range(1, 4)]}")
  print("max_abs row4-groups x col128-blocks=", [[float(delta[r:r+4, c:c+128].max())
        for c in range(0, args.n, 128)] for r in range(0, args.m, 4)])
  if not np.allclose(expected, got, rtol=2e-2, atol=2e-2): raise SystemExit(1)


if __name__ == "__main__": main()
