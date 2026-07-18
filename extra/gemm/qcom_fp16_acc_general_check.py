#!/usr/bin/env python3
"""Full random-matrix correctness check for the fast A630 FP16-accumulate GEMM."""
import ctypes, importlib.util, os, struct

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from tinygrad.helpers import ceildiv
from tinygrad.runtime.ops_qcom import dcache_flush
from extra.gemm.ir3asm import disasm, get_envelope, inject
from extra.gemm.qcom_gemm import patch_kernel

if module_path := os.getenv("QCOM_INTENSITY_MODULE"):
  spec = importlib.util.spec_from_file_location("qcom_intensity_snapshot", module_path)
  if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {module_path}")
  q = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(q)
else:
  from extra.gemm import qcom_intensity_gemm as q


def install_random_safe_store() -> None:
  """Replace the all-ones-only repeated move in the preserved fast kernel's epilogue."""
  original_mov_h = q.MOV_H
  def random_safe_mov_h(dst, src, rpt=0, r=False):
    # In this path every repeated MOV broadcasts the zero accumulator seed. On A630 the
    # relative-source repeat also walks the source, so use an immediate vector fill instead.
    return q.MOV_H_IMM(dst, 0, rpt=rpt) if rpt else original_mov_h(dst, src, r=r)
  q.MOV_H = random_safe_mov_h
  hand_addr = [bytes.fromhex(x) for x in (
    "1c000a200000d04e 1d0002200100d046 0000000000100000 000061100201b843 000060100600b043 0000010000003042 "
    "0100020002013842 0100060001003042 000001200000d046 020001200200d046 030001200600d046 010001200700d046 "
    "0000000003401520 0000000000000000 0600000001401520 0700000000401520 0000000000100000 5010030008001042 "
    "501002000a001042 501001000c001042 501000000e001042 0000000000100000 0800501010009042 03001f201100f046 "
    "0a00501012009042 02001f201300f046 0c00501014009042 01001f201500f046 0e00501016009042 00001f201700f046 "
    "0000000000100000 5110104009808867 511012400b808967 511014400d808a67 519016400f888b67").split()]
  gap = int(os.getenv("STORE_GAP", "0"))
  hand_stores = []
  for row, addr in enumerate(("r2.x", "r2.z", "r3.x", "r3.z")):
    hand_stores.append(q.STG_F16(addr, row*4))
    if row != 3: hand_stores.append(q.NOP(rpt=gap))
  def emit(instrs, acc0, ncols, *_args, **_kwargs):
    for col in range(ncols):
      if col: instrs.append(q.ADD_S("r7.y", "r7.y", 32))
      instrs += hand_addr
      for row in range(4):
        for lane in range(4): instrs.append(q.MOV_H(row*4+lane, acc0+(row*ncols+col)*4+lane))
      instrs += hand_stores
  q.emit_hand4_stores = emit


def upload(x: np.ndarray, dtype) -> Buffer:
  ret = Buffer("QCOM", x.size, dtype).allocate()
  raw = memoryview(np.ascontiguousarray(x)).cast("B")
  ret.copyin(raw) if hasattr(ret, "copyin") else Device[ret.device].allocator._copyin(ret._buf, raw)
  ptr = ret._buf.cpu_view().addr
  dcache_flush().fxn(ctypes.c_uint64(ptr & ~63), ceildiv(ptr + ret.nbytes - (ptr & ~63), 64))
  return ret


def main() -> None:
  m, n, k = (int(os.getenv(name, "1024")) for name in ("M", "N", "K"))
  seed, threads = int(os.getenv("SEED", "901")), int(os.getenv("THREADS", "128"))
  ncols = int(os.getenv("NCOLS", "4"))
  if m % 16 or n % (128*ncols) or k % 16: raise ValueError("M, N, K must divide the selected tile")
  if threads not in (64, 128, 256): raise ValueError("THREADS must be 64, 128, or 256")
  rng = np.random.default_rng(seed)
  a_np = (np.eye(m, k, dtype=np.float16) if os.getenv("PATTERN") == "identity" else
          (rng.standard_normal((m, k), dtype=np.float32)*np.float32(0.05)).astype(np.float16))
  b_np = (rng.standard_normal((k, n), dtype=np.float32)*np.float32(0.05)).astype(np.float16)

  q.M, q.N, q.K, q.K4 = m, n, k, k//4
  install_random_safe_store()
  dev = Device["QCOM"]
  if os.getenv("COMPILER_DIRECT"):
    compiler_partial = bool(int(os.getenv("COMPILER_PARTIAL", "0")))
    compiler_image = bool(int(os.getenv("COMPILER_IMAGE", "0")))
    compiler_src = (q.make_direct_image_donor_src(ncols, threads) if compiler_image else
                    q.make_donor_src(ncols, threads) if compiler_partial else q.make_direct_donor_src(ncols, threads))
    lib = dev.compiler.compile_cached(compiler_src)
    if os.getenv("PATCH_COMPILER", "1") != "0":
      lib = patch_kernel(lib, os.getenv("PATCH_SYNC", "1") != "0", os.getenv("MERGE_PAIRS", "1") != "0", int(os.getenv("MAX_GROUPS", "-1")))
    if os.getenv("PRINT_ASM"):
      io, sz = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
      print(disasm(lib[io:io+sz]))
    loop_instrs = -1
  else:
    image_output = bool(int(os.getenv("IMAGE_OUTPUT", "0")))
    env_src = q.make_direct_image_donor_src(4, threads) if image_output else q.make_donor_src(4, threads)
    env, io, sz, ro = get_envelope(dev, env_src)
    low_a = bool(int(os.getenv("LOW_A", "0")))
    high_inputs = bool(int(os.getenv("HIGH_INPUTS", "0")))
    high_a_only = bool(int(os.getenv("HIGH_A_ONLY", "0")))
    extra = ({"high_inputs": high_inputs, "high_a_only": high_a_only}
             if "high_inputs" in __import__("inspect").signature(q.build_4xn_shader).parameters else {})
    extra["serial_b_cols"] = bool(int(os.getenv("SERIAL_B_COLS", "0")))
    k_unroll = int(os.getenv("K_UNROLL", "4"))
    if image_output:
      persistent = bool(int(os.getenv("PERSISTENT", "0")))
      advanced = ncols == 4
      shader, loop_instrs = q.build_4xn_shader(dev, threads, ncols=ncols, direct=True, b_first=advanced, compact_acc=True,
        stable_bx=advanced, stable_ay=advanced, low_a_coords=low_a, inc_coords=persistent and advanced,
        persistent_coords=persistent and advanced, k_unroll=k_unroll,
        alu_order="row_col_kk", first_sync_only=bool(int(os.getenv("FIRST_SYNC_ONLY", "1"))),
        row_sync=bool(int(os.getenv("ROW_SYNC", "0"))), coord_delay=int(os.getenv("COORD_DELAY", "-1")),
        image_store=True, preserve_coords=bool(int(os.getenv("PRESERVE_COORDS", "1"))),
        safe_b_y=bool(int(os.getenv("SAFE_B_Y", "0"))), sync_b_y=bool(int(os.getenv("SYNC_B_Y", "0"))),
        separate_b_coords=bool(int(os.getenv("SEPARATE_B_COORDS", "0"))), high_b_coords=bool(int(os.getenv("HIGH_B_COORDS", "0"))),
        persistent_b_x=bool(int(os.getenv("PERSISTENT_B_X", "0"))), **extra)
    else:
      advanced = ncols == 4
      shader, loop_instrs = q.build_4xn_shader(dev, threads, ncols=ncols, direct=True, b_first=advanced, compact_acc=True,
        stable_bx=advanced, stable_ay=advanced, low_a_coords=low_a, inc_coords=advanced, persistent_coords=advanced, k_unroll=k_unroll,
        alu_order="row_col_kk", first_sync_only=bool(int(os.getenv("FIRST_SYNC_ONLY", "1"))),
        row_sync=bool(int(os.getenv("ROW_SYNC", "0"))), coord_delay=int(os.getenv("COORD_DELAY", "-1")),
        safe_b_y=bool(int(os.getenv("SAFE_B_Y", "0"))), sync_b_y=bool(int(os.getenv("SYNC_B_Y", "0"))),
        separate_b_coords=bool(int(os.getenv("SEPARATE_B_COORDS", "0"))), high_b_coords=bool(int(os.getenv("HIGH_B_COORDS", "0"))),
        persistent_b_x=bool(int(os.getenv("PERSISTENT_B_X", "0"))), **extra)
    lib = inject(env, io, sz, ro, shader, fregs=13 if bool(int(os.getenv("PERSISTENT_B_X", "0"))) else 8 if low_a else 10,
                 hregs=48 if high_inputs else 36 if high_a_only else 28)
    asm = disasm(shader)
    if asm.count("mad.f16") != 16*ncols*k_unroll or asm.count("mad.f32") != 0:
      raise RuntimeError("unexpected accumulator instruction mix")

  a, b = upload(a_np, dtypes.half), upload(b_np, dtypes.half)
  image_output = (bool(int(os.getenv("IMAGE_OUTPUT", "0"))) and not os.getenv("COMPILER_DIRECT")) or \
                 bool(int(os.getenv("COMPILER_IMAGE", "0")))
  c_np, c_dtype = (np.zeros((m, n), np.float32), dtypes.float) if image_output else (np.zeros((m, n), np.float16), dtypes.half)
  c = upload(c_np, c_dtype)
  if image_output:
    # The image envelope declares C first so QCOMArgsState assigns its sole IBO to C,
    # followed by A/B in texture slots 0/1 as expected by the injected prologue.
    prg = dev.runtime("gemm_h", lib, buf_dtypes=[
      ((0, dtypes.float, (m, n//4, 4)),), ((0, dtypes.half, (m, k//4, 4)),), ((0, dtypes.half, (k, n//4, 4)),)])
    args = (c._buf, a._buf, b._buf)
  else:
    prg = dev.runtime("gemm_h", lib, buf_dtypes=[
      ((0, dtypes.half, (m, k//4, 4)),), ((0, dtypes.half, (k, n//4, 4)),), ((0, dtypes.half, None),)])
    args = (a._buf, b._buf, c._buf)
  gs, ls = (n//(128*ncols), m//((threads//32)*4), 1), (threads, 1, 1)
  for _ in range(int(os.getenv("WARM", "3"))): prg(*args, global_size=gs, local_size=ls, wait=True)
  times = [prg(*args, global_size=gs, local_size=ls, wait=True) for _ in range(int(os.getenv("RUNS", "10")))]

  got = np.empty((m, n), c_np.dtype)
  raw = memoryview(got).cast("B")
  c.copyout(raw) if hasattr(c, "copyout") else Device[c.device].allocator._copyout(raw, c._buf)
  expected = a_np.astype(np.float32) @ b_np.astype(np.float32)
  delta = np.abs(got.astype(np.float32)-expected)
  rtol, atol = float(os.getenv("RTOL", "0.02")), float(os.getenv("ATOL", "0.02"))
  bad = ~np.isclose(got, expected, rtol=rtol, atol=atol)
  if os.getenv("SHOW_VALUES"):
    print("bad_by_row_mod4", [int(bad[r::4].sum()) for r in range(4)])
    print("bad_by_col_block", [int(bad[:, c:c+128].sum()) for c in range(0, n, 128)])
    for row in range(4):
      print(f"row={row} got={got[row, :32].astype(np.float32).tolist()}")
      print(f"row={row} exp={expected[row, :32].tolist()}")
    if os.getenv("PATTERN") == "identity":
      for row in range(16):
        mse = np.mean((expected[:, :128]-got[row, :128])**2, axis=1)
        print(f"identity_row={row} nearest_expected_row={int(np.argmin(mse))} mse={float(mse.min()):.9g}")
  best = min(x for x in times if x is not None)
  print(f"shape={m}x{n}x{k} inputs=fp16 accumulate=fp16 elapsed_ms={best*1e3:.3f} "
        f"gflops={2*m*n*k/best/1e9:.1f} outputs={m*n} bad_count={int(bad.sum())} "
        f"max_abs={float(delta.max()):.9g} mean_abs={float(delta.mean()):.9g} "
        f"rtol={rtol:g} atol={atol:g} allclose={not bool(bad.any())} loop_instrs={loop_instrs}")
  if bad.any(): raise SystemExit(1)


if __name__ == "__main__": main()
