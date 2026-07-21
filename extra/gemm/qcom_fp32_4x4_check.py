#!/usr/bin/env python3
"""Randomized oracle for the compact FP32-accumulating 4x4 QCOM GEMM."""
import argparse

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import get_envelope, inject


def upload(x:np.ndarray, dtype) -> Buffer:
  ret = Buffer("QCOM", x.size, dtype).allocate()
  ret.copyin(memoryview(np.ascontiguousarray(x)).cast("B"))
  return ret


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--m", type=int, default=128)
  ap.add_argument("--n", type=int, default=1536)
  ap.add_argument("--k", type=int, default=384)
  ap.add_argument("--stride", type=int, default=0)
  ap.add_argument("--threads", type=int, default=128, choices=(64, 128, 256))
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--coord-delay", type=int, default=4)
  ap.add_argument("--first-wait-only", action="store_true")
  ap.add_argument("--batch-coords", action="store_true")
  ap.add_argument("--quad-map", action="store_true", help="map each quad to one output column")
  ap.add_argument("--quad-b", action="store_true", help="load B in one quad lane and broadcast it")
  ap.add_argument("--quad-b-load-all", action="store_true", help="load B in all lanes before broadcasting lane zero")
  ap.add_argument("--quad-b-shfl-mode", type=int, default=0, help="use scalar relative shuffles instead of vector quad broadcast")
  ap.add_argument("--post-constant", action="store_true", help="replace accumulators with 1024 before storing")
  ap.add_argument("--float-inputs", action="store_true", help="store both sampled inputs as float32 images")
  args = ap.parse_args()
  rng = np.random.default_rng(args.seed)
  input_np_dtype = np.float32 if args.float_inputs else np.float16
  input_dtype = dtypes.float if args.float_inputs else dtypes.half
  a_np = (rng.standard_normal((args.m, args.k))*0.05).astype(input_np_dtype)
  b_np = (rng.standard_normal((args.k, args.n))*0.05).astype(input_np_dtype)
  stride = args.stride or (2048 if args.n > 1024 else 1024)
  q.M, q.N, q.K, q.K4 = args.m, stride, args.k, args.k//4
  dev = Device["QCOM"]
  env, io, sz, ro = get_envelope(dev, q.make_direct_donor_src_fp32(1, args.threads))
  shader, hregs, fregs, _ = q.build_4x4_fp32_compact_preload_shader(
    dev, args.threads, coord_delay=args.coord_delay, sampler_per_texture=True, post_constant=args.post_constant,
    batch_coords=args.batch_coords, first_coord_wait_only=args.first_wait_only,
    quad_map=args.quad_map, quad_b=args.quad_b, quad_b_load_all=args.quad_b_load_all,
    quad_b_shfl_mode=args.quad_b_shfl_mode)
  lib = inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs)
  a, b = upload(a_np, input_dtype), upload(b_np, input_dtype)
  c = upload(np.zeros(args.m*stride, np.float32), dtypes.float)
  specs = [((0, input_dtype, (args.m, args.k//4, 4)),),
           ((1, input_dtype, (args.k, args.n//4, 4)),), ((2, dtypes.float, (args.m*stride,)),)]
  program = dev.runtime("gemm_h", lib, buf_dtypes=specs)
  tile_m = (args.threads//32)*4
  times = [program(a._buf, b._buf, c._buf, global_size=(args.n//128, args.m//tile_m, 1),
                   local_size=(args.threads, 1, 1), wait=True)*1e3 for _ in range(10)]
  got_storage = np.empty((args.m, stride), np.float32)
  c.copyout(memoryview(got_storage).cast("B"))
  got = got_storage[:, :args.n]
  expected = np.full((args.m,args.n), 1024, np.float32) if args.post_constant else a_np.astype(np.float32) @ b_np.astype(np.float32)
  delta = np.abs(expected-got)
  worst = np.unravel_index(int(delta.argmax()), delta.shape)
  passed = bool(np.allclose(expected, got, rtol=1e-4, atol=1e-4))
  print(f"ms={min(times):.4f} max_abs={float(delta.max()):.9g} mean_abs={float(delta.mean()):.9g} "
        f"worst={worst} allclose={passed}")
  if not passed:
    nz = np.argwhere(got_storage != 0)
    print("storage_nonzero=", int(nz.shape[0]), "first=", nz[:32].tolist())
    row_cost = np.mean(np.abs(expected[:,None,:]-got[None,:,:]), axis=2)
    print("best_expected_row_for_got=", [(int(j), int(np.argmin(row_cost[:,j])), float(np.min(row_cost[:,j]))) for j in range(min(32,args.m))])
    print("row0_expected=", expected[0,:16].tolist())
    print("row0_got=", got[0,:16].tolist())
  if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
