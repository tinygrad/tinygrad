#!/usr/bin/env python3
"""Full-random oracle and timer for the streamed wide FP32-accumulate GEMM."""
import os

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import get_envelope, inject


def upload(x: np.ndarray, dtype) -> Buffer:
  raw = np.ascontiguousarray(x)
  return Buffer("QCOM", raw.size, dtype, initial_value=memoryview(raw).cast("B").tobytes())


def main() -> None:
  m, n, k = (int(os.getenv(x, d)) for x, d in (("M", "256"), ("N", "512"), ("K", "1024")))
  batch = int(os.getenv("BATCH", "1"))
  batch_y = batch > 1 and bool(int(os.getenv("BATCH_Y", "1")))
  ncols, threads = int(os.getenv("NCOLS", "4")), int(os.getenv("THREADS", "128"))
  custom_rows = int(os.getenv("CUSTOM_ROWS", "0"))
  stride, seed = int(os.getenv("STRIDE", str(n))), int(os.getenv("SEED", "0"))
  k_start = int(os.getenv("K_START", "0"))
  rows8, square8 = bool(int(os.getenv("ROWS8", "0"))), bool(int(os.getenv("SQUARE8", "0")))
  quad_a = bool(int(os.getenv("QUAD_A", "0")))
  quad_split = bool(int(os.getenv("QUAD_SPLIT", "0")))
  rows8 = rows8 or square8
  tile_m = (threads//32)*(custom_rows or (8 if rows8 else 4))
  tile_n = 32*ncols if quad_split else 128*ncols
  assert m % tile_m == 0 and n % tile_n == 0 and k % 4 == 0
  rng = np.random.default_rng(seed)
  a_np = (rng.standard_normal((batch, m, k))*0.05).astype(np.float16)
  b_np = (rng.standard_normal((batch, k, n))*0.05).astype(np.float16)
  if os.getenv("PATTERN") == "ones":
    a_np.fill(1)
    b_np.fill(1)
  q.M, q.N, q.K, q.K4 = m, stride, k, k//4
  dev = Device["QCOM"]
  rotate_buffer = bool(int(os.getenv("ROTATE_BUFFER", "0")))
  swap_groups = bool(int(os.getenv("SWAP_GROUPS", "0")))
  column_z = bool(int(os.getenv("COLUMN_Z", "0")))
  if column_z and (batch != 1 or swap_groups): raise ValueError("COLUMN_Z requires BATCH=1 and SWAP_GROUPS=0")
  output_half = bool(int(os.getenv("OUTPUT_HALF", "0")))
  int8_b = bool(int(os.getenv("INT8_B", "0")))
  int8_a = bool(int(os.getenv("INT8_A", "0")))
  env_src = q.make_direct_donor_src_fp32(4, threads) if rotate_buffer else \
            q.make_direct_image_donor_src(ncols, threads, swap_groups=swap_groups)
  env, io, sz, ro = get_envelope(dev, env_src)
  if quad_split:
    if ncols != 2: raise ValueError("QUAD_SPLIT=1 requires NCOLS=2")
    shader, hregs, fregs, loop_instrs = q.build_4x8_fp32_quad_splitk_shader(
      dev, threads, store_gap=int(os.getenv("STORE_GAP", "16")), no_reduce=bool(int(os.getenv("NO_REDUCE", "0"))),
      k_count=int(os.getenv("K_COUNT", str(k//4))))
  elif custom_rows:
    if ncols != 2: raise ValueError("CUSTOM_ROWS requires NCOLS=2")
    shader, hregs, fregs, loop_instrs = q.build_rx8_fp32_shader(
      dev, threads, rows=custom_rows, store_gap=int(os.getenv("STORE_GAP", "16")))
  elif quad_a:
    if ncols != 2: raise ValueError("QUAD_A=1 requires NCOLS=2")
    shader, hregs, fregs, loop_instrs = q.build_4x8_fp32_quad_a_shader(
      dev, threads, store_gap=int(os.getenv("STORE_GAP", "16")))
  elif square8:
    if ncols != 2: raise ValueError("SQUARE8=1 requires NCOLS=2")
    shader, hregs, fregs, loop_instrs = q.build_8x8_fp32_shader(
      dev, threads, store_gap=int(os.getenv("STORE_GAP", "16")))
  elif rows8:
    if ncols != 1: raise ValueError("ROWS8=1 requires NCOLS=1")
    shader, hregs, fregs, loop_instrs = q.build_8x4_fp32_shader(
      dev, threads, store_gap=int(os.getenv("STORE_GAP", "16")))
  elif bool(int(os.getenv("WAKSMAN", "0"))):
    if ncols != 2: raise ValueError("WAKSMAN=1 requires NCOLS=2")
    shader, hregs, fregs, loop_instrs = q.build_4x8_waksman_fp32_shader(
      dev, threads, store_gap=int(os.getenv("STORE_GAP", "16")), no_q=bool(int(os.getenv("WAKSMAN_NO_Q", "0"))))
  elif bool(int(os.getenv("ROTATE", "0"))):
    if ncols != 2: raise ValueError("ROTATE=1 requires NCOLS=2")
    shader, hregs, fregs, loop_instrs = q.build_4x8_fp32_rotate_shader(
      dev, threads, store_gap=int(os.getenv("STORE_GAP", "16")),
      post_constant=bool(int(os.getenv("POST_CONSTANT", "0"))), image_store=not rotate_buffer,
      k_count=int(os.getenv("K_COUNT", str(k//4))), batch_stride=m if batch > 1 else 0, batch_from_row=batch_y, k_start=k_start,
      k_unroll=int(os.getenv("K_UNROLL", "3")), swap_groups=swap_groups, col_from_z=column_z)
  else:
    shader, hregs, fregs, loop_instrs = q.build_4xn_fp32_stream_shader(
      dev, threads, ncols=ncols, coord_delay=int(os.getenv("COORD_DELAY", "-1")),
      sync_each_col=bool(int(os.getenv("SYNC_EACH_COL", "1"))), store_gap=int(os.getenv("STORE_GAP", "16")),
      post_constant=bool(int(os.getenv("POST_CONSTANT", "0"))), pipeline_b=bool(int(os.getenv("PIPELINE_B", "0"))),
      component_stream=bool(int(os.getenv("COMPONENT_STREAM", "0"))),
      component_sync_kk=int(os.getenv("COMPONENT_SYNC_KK", "0")))
  lib = inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs, mergedregs=False)
  a_upload = a_np.reshape(batch*m//4, 4, k).transpose(0, 2, 1).copy() if quad_split else a_np
  if int8_a:
    a_upload = np.clip(np.rint(a_upload.astype(np.float32)*127), -127, 127).astype(np.int8)
    a_np = a_upload.astype(np.float32)/127
  if int8_b:
    b_upload = np.clip(np.rint(b_np.astype(np.float32)*127), -127, 127).astype(np.int8)
    b_np = b_upload.astype(np.float32)/127
  else: b_upload = b_np
  a, b = upload(a_upload, dtypes.int8 if int8_a else dtypes.half), upload(b_upload, dtypes.int8 if int8_b else dtypes.half)
  c = upload(np.zeros((batch*m, stride), np.float16 if output_half else np.float32), dtypes.half if output_half else dtypes.float)
  specs = ([((0, dtypes.half, (m, k//4, 4)),), ((1, dtypes.half, (k, n//4, 4)),),
            ((0, dtypes.float, None),)] if rotate_buffer else
           [((0, dtypes.half if output_half else dtypes.float, (batch*m, stride//4, 4)),),
            ((0, dtypes.int8 if int8_a else dtypes.half,
              (batch*m//4, k, 4) if quad_split else (batch*m, k//4, 4)),),
            ((1, dtypes.int8 if int8_b else dtypes.half, (batch*k, n//4, 4)),)])
  prg = dev.runtime("gemm_h", lib, buf_dtypes=specs)
  runs = int(os.getenv("BENCH_RUNS", "10"))
  call_bufs = (a._buf, b._buf, c._buf) if rotate_buffer else (c._buf, a._buf, b._buf)
  launch_x, launch_y = ((batch*m//tile_m if batch_y else m//tile_m), n//tile_n) if swap_groups else \
                       (1 if column_z else n//tile_n, batch*m//tile_m if batch_y else m//tile_m)
  launch_z = n//tile_n if column_z else 1 if batch_y else batch
  times = [prg(*call_bufs, global_size=(launch_x, launch_y, launch_z),
               local_size=(threads, 1, 1), wait=True) for _ in range(runs)]
  got_storage = np.empty((batch*m, stride), np.float16 if output_half else np.float32)
  got_storage.reshape(-1)[:] = c.numpy().reshape(-1)
  got = got_storage[:, :n].reshape(batch, m, n)
  expected = np.full((batch, m, n), 1024, np.float32) if bool(int(os.getenv("POST_CONSTANT", "0"))) else \
             a_np[:, :, k_start*4:(k_start+int(os.getenv("K_COUNT", str(k//4))))*4].astype(np.float32) @ \
             b_np[:, k_start*4:(k_start+int(os.getenv("K_COUNT", str(k//4))))*4].astype(np.float32)
  delta = np.abs(got-expected)
  passed = bool(np.allclose(got, expected, rtol=1e-4, atol=1e-4))
  elapsed = min(times)
  bad = int(np.count_nonzero(~np.isclose(got, expected, rtol=1e-4, atol=1e-4)))
  print(f"shape={batch}x{m}x{n}x{k} inputs=fp16 accumulate=fp32 elapsed_ms={elapsed*1e3:.3f} "
        f"gflops={batch*2*m*n*k/elapsed/1e9:.1f} fregs={fregs} loop_instrs={loop_instrs} "
        f"max_abs={float(delta.max()):.9g} mean_abs={float(delta.mean()):.9g} allclose={passed} bad_count={bad}")
  if not passed:
    where = np.argwhere(~np.isclose(got, expected, rtol=1e-4, atol=1e-4))
    print("bad_head=", where[:32].tolist(), "got_head=", got.reshape(-1)[:32].tolist(),
          "expected_head=", expected.reshape(-1)[:32].tolist())
    raise SystemExit(1)


if __name__ == "__main__": main()
