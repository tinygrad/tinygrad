#!/usr/bin/env python3
"""Random-matrix oracle for hand-assembled QCOM FP32-accumulating GEMMs."""
import argparse

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm import qcom_8x4_gemm as q8
from extra.gemm.ir3asm import get_envelope, inject


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--m", type=int, default=192)
  parser.add_argument("--n", type=int, default=512)
  parser.add_argument("--k", type=int, default=768)
  parser.add_argument("--stride", type=int, default=0)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--threads", type=int, default=128, choices=(64, 128))
  parser.add_argument("--preload-b", action="store_true")
  parser.add_argument("--batch-coords", action="store_true")
  parser.add_argument("--hand-store", action="store_true")
  parser.add_argument("--interleaved-a", action="store_true")
  parser.add_argument("--no-store", action="store_true", help="profile compute only; skip output validation")
  parser.add_argument("--compiler", action="store_true", help="run the unmodified compiler-generated 4x8 kernel")
  parser.add_argument("--pipeline", action="store_true", help="run the double-buffered hand 4x8 kernel")
  parser.add_argument("--alu-order", default="kk_row_col")
  parser.add_argument("--coord-delay", type=int, default=-1)
  parser.add_argument("--identity-b", action="store_true", help="make the output expose the first N activation columns")
  parser.add_argument("--float-inputs", action="store_true", help="store both sampled inputs as float32 images")
  parser.add_argument("--eight-row", action="store_true", help="test the scalar FP32 8x4 kernel")
  parser.add_argument("--compact-preload", action="store_true", help="test the compact FP32 4x4 preload kernel")
  parser.add_argument("--first-wait-only", action="store_true")
  parser.add_argument("--quad-map", action="store_true")
  parser.add_argument("--quad-b", action="store_true")
  parser.add_argument("--quad-b-load-all", action="store_true")
  args = parser.parse_args()
  n_tile = 128 if args.eight_row or args.compact_preload else 256
  tile_m = (args.threads//32) * (8 if args.eight_row else 4)
  assert args.m % tile_m == 0 and args.n % n_tile == 0 and args.k % 4 == 0

  rng = np.random.default_rng(args.seed)
  input_np_dtype = np.float32 if args.float_inputs else np.float16
  input_dtype = dtypes.float if args.float_inputs else dtypes.half
  a_np = (rng.standard_normal((args.m, args.k))*0.1).astype(input_np_dtype)
  b_np = (rng.standard_normal((args.k, args.n))*0.1).astype(input_np_dtype)
  if args.identity_b:
    b_np.fill(0)
    np.fill_diagonal(b_np, np.float16(1))
  stride = args.stride or (2048 if args.n > 1024 else 1024)
  q.M, q.N, q.K, q.K4 = args.m, stride, args.k, args.k//4
  q8.M, q8.N, q8.K, q8.K4 = args.m, stride, args.k, args.k//4
  dev = Device["QCOM"]
  donor_src = q8.make_donor_src8_fp32(1, args.threads) if args.eight_row else q.make_direct_donor_src_fp32(2, args.threads)
  envelope, image_offset, image_size, register_offset = get_envelope(dev, donor_src)
  if args.compiler:
    lib = bytes(envelope)
  else:
    if args.eight_row:
      shader, hregs, fregs, _ = q8.build_8x8_fp32_shader(
        dev, args.threads, ncols=1, b_coord_delay=args.coord_delay, alu_order=args.alu_order)
    elif args.compact_preload:
      shader, hregs, fregs, _ = q.build_4x4_fp32_compact_preload_shader(
        dev, args.threads, coord_delay=args.coord_delay, sampler_per_texture=True,
        batch_coords=args.batch_coords, quad_map=args.quad_map, quad_b=args.quad_b,
        quad_b_load_all=args.quad_b_load_all, first_coord_wait_only=args.first_wait_only)
    elif args.pipeline:
      shader, hregs, fregs, _ = q.build_4x8_fp32_pipeline_shader(
        dev, args.threads, coord_delay=args.coord_delay, sampler_per_texture=True, no_store=args.no_store)
    else:
      shader, hregs, fregs, _ = q.build_4x8_fp32_low_shader(
        dev, args.threads, coord_delay=args.coord_delay, sampler_per_texture=True, alu_order=args.alu_order,
        preload_b=args.preload_b, batch_coords=args.batch_coords, hand_store=args.hand_store,
        interleaved_a=args.interleaved_a, no_store=args.no_store)
    lib = inject(envelope, image_offset, image_size, register_offset, shader, fregs=fregs, hregs=hregs)

  a_upload = a_np.reshape(args.m//4, 4, args.k).transpose(0, 2, 1).copy() if args.interleaved_a else a_np
  a = Buffer("QCOM", a_upload.size, input_dtype, initial_value=memoryview(a_upload).cast("B").tobytes())
  b = Buffer("QCOM", b_np.size, input_dtype, initial_value=memoryview(b_np).cast("B").tobytes())
  c = Buffer("QCOM", args.m*stride, dtypes.float,
             initial_value=memoryview(np.zeros(args.m*stride, dtype=np.float32)).cast("B").tobytes())
  a_shape = (args.m//4, args.k, 4) if args.interleaved_a else (args.m, args.k//4, 4)
  specs = [((0, input_dtype, a_shape),), ((0, input_dtype, (args.k, args.n//4, 4)),),
           ((0, dtypes.float, None),)]
  program = dev.runtime("gemm_h" if args.eight_row else "gemm_f", lib, buf_dtypes=specs)
  for _ in range(3):
    elapsed = program(a._buf, b._buf, c._buf, global_size=(args.n//n_tile, args.m//tile_m, 1),
                      local_size=(args.threads, 1, 1), wait=True)
  if args.no_store:
    print(f"elapsed_ms={elapsed*1e3:.3f} compute_only=True")
    return
  got_flat = np.empty(args.m*stride, dtype=np.float32)
  got_flat[:] = c.numpy()
  got = got_flat.reshape(args.m, stride)[:, :args.n]
  expected = a_np.astype(np.float32) @ b_np.astype(np.float32)
  delta = np.abs(expected-got)
  worst = np.unravel_index(np.argmax(delta), delta.shape)
  passed = bool(np.allclose(expected, got, rtol=1e-4, atol=1e-4))
  print(f"elapsed_ms={elapsed*1e3:.3f} max_abs={float(delta.max()):.9g} mean_abs={float(delta.mean()):.9g} "
        f"worst={worst} expected={expected[worst]!r} got={got[worst]!r} allclose={passed}")
  print("max_abs row16 x col256=", [[float(delta[r:r+16, c:c+256].max()) for c in range(0, args.n, 256)]
                                      for r in range(0, args.m, 16)])
  print(f"nonzero={np.count_nonzero(got)/got.size:.3f} got_row0={got[0, :8].tolist()} expected_row0={expected[0, :8].tolist()}")
  if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
