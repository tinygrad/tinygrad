#!/usr/bin/env python3
"""Random-matrix oracle for the hand IR3 4x16 FP16 GEMM."""
import os, struct

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_intensity_gemm as q
from extra.gemm.ir3asm import disasm, get_envelope, inject


def upload(x: np.ndarray) -> Buffer:
  ret = Buffer("QCOM", x.size, dtypes.half).allocate()
  ret.copyin(memoryview(np.ascontiguousarray(x)).cast("B"))
  return ret


def strip_redundant_mad_sy(lib: bytes) -> bytes:
  ret = bytearray(lib)
  io, sz = struct.unpack_from('<I', ret, 0xc0)[0], struct.unpack_from('<I', ret, 0x100)[0]
  seen = False
  for off in range(io, io+sz, 8):
    hi = struct.unpack_from('<I', ret, off+4)[0]
    if (hi >> 24) == 0x73:
      if seen: struct.pack_into('<I', ret, off+4, (hi & 0x0fffffff) | 0x60000000)
      else: seen = True
  return bytes(ret)


def restore_all_mad_sy(lib: bytes) -> bytes:
  ret = bytearray(lib)
  io, sz = struct.unpack_from('<I', ret, 0xc0)[0], struct.unpack_from('<I', ret, 0x100)[0]
  for off in range(io, io+sz, 8):
    hi = struct.unpack_from('<I', ret, off+4)[0]
    if (hi >> 24) == 0x63: struct.pack_into('<I', ret, off+4, (hi & 0x0fffffff) | 0x70000000)
  return bytes(ret)


def restore_original_mad_sy(lib: bytes, original: bytes) -> bytes:
  ret = bytearray(lib)
  io, sz = struct.unpack_from('<I', ret, 0xc0)[0], struct.unpack_from('<I', ret, 0x100)[0]
  for off in range(io, io+sz, 8):
    hi = struct.unpack_from('<I', ret, off+4)[0]
    old_hi = struct.unpack_from('<I', original, off+4)[0]
    if (hi >> 24) == 0x63 and (old_hi >> 24) == 0x73:
      struct.pack_into('<I', ret, off+4, (hi & 0x0fffffff) | 0x70000000)
  return bytes(ret)


def main() -> None:
  m, n, k = int(os.getenv("M", "128")), int(os.getenv("N", "1024")), int(os.getenv("K", "384"))
  stride = int(os.getenv("STRIDE", str(n)))
  ncols = int(os.getenv("NCOLS", "4"))
  threads = int(os.getenv("THREADS", "128"))
  rng = np.random.default_rng(int(os.getenv("SEED", "4")))
  a = (rng.standard_normal((m, k))*0.05).astype(np.float16)
  b = (rng.standard_normal((k, n))*0.05).astype(np.float16)
  if pattern := os.getenv("PATTERN", ""):
    a.fill(0)
    b.fill(0)
    if pattern == "row":
      a[:, 0] = np.arange(1, m+1)
      b[0, :] = 1
    elif pattern == "col":
      a[:, 0] = 1
      b[0, :] = (np.arange(n) % 251) + 1
    elif pattern.startswith("k"):
      kk = int(pattern[1:])
      a[:, kk] = np.arange(1, m+1)
      b[kk, :] = 1
    else: raise ValueError(f"unknown PATTERN={pattern!r}")
  q.M, q.N, q.K, q.K4 = m, stride, k, k//4
  dev = Device["QCOM"]
  compiler = bool(int(os.getenv("COMPILER", "0")))
  env_ncols = int(os.getenv("ENV_NCOLS", str(ncols)))
  direct_env = compiler or bool(int(os.getenv("ENV_DIRECT", "0")))
  image_store = bool(int(os.getenv("IMAGE_STORE", "0")))
  output_float = bool(int(os.getenv("OUTPUT_FLOAT", "0")))
  dynamic_splits = int(os.getenv("DYNAMIC_SPLIT", "0"))
  env_src = q.make_direct_image_donor_src(env_ncols, threads) if image_store else \
            q.make_direct_donor_src(env_ncols if direct_env else ncols, threads) if direct_env else q.make_donor_src(env_ncols, threads)
  env, io, sz, ro = get_envelope(dev, env_src)
  fast = bool(int(os.getenv("FAST", "1")))
  preserve_coords = bool(int(os.getenv("PRESERVE_COORDS", "0")))
  high_inputs = bool(int(os.getenv("HIGH_INPUTS", "0")))
  high_store = bool(int(os.getenv("HIGH_STORE", "0")))
  low_a = bool(int(os.getenv("LOW_A", "0")))
  inc = bool(int(os.getenv("INC", str(int(fast and not preserve_coords)))))
  persistent = bool(int(os.getenv("PERSISTENT", str(int(fast and inc and not preserve_coords)))))
  unroll = int(os.getenv("K_UNROLL", str(4 if k % 16 == 0 else 1)))
  k_count = int(os.getenv("K_COUNT", str(k//4)))
  if compiler:
    patch_mode = os.getenv("PATCH_COMPILER", "0")
    if patch_mode == "sync": lib = strip_redundant_mad_sy(env)
    elif patch_mode != "0":
      from extra.gemm.qcom_gemm import patch_kernel
      lib = patch_kernel(env)
      if patch_mode == "rpt": lib = restore_all_mad_sy(lib)
      elif patch_mode == "original": lib = restore_original_mad_sy(lib, env)
    else: lib = env
    shader = bytes(env[io:io+sz])
  else:
    safe_store = bool(int(os.getenv("SAFE_STORE", "0")))
    compact = bool(int(os.getenv("COMPACT", str(int(not safe_store)))))
    isolated = bool(int(os.getenv("ISOLATED", "0")))
    shader, _ = q.build_4x16_isolated_shader(dev, threads, k_unroll=unroll) if isolated else q.build_4xn_shader(
      dev, threads, ncols=ncols, direct=True, compact_acc=compact,
      store_constant=bool(int(os.getenv("STORE_CONSTANT", "0"))),
      donor_store=bool(int(os.getenv("DONOR_STORE", "0"))),
      native_store=bool(int(os.getenv("NATIVE_STORE", "0"))),
      safe_store=safe_store,
      linear_store=bool(int(os.getenv("LINEAR_STORE", "0"))),
      image_store=image_store,
      preserve_coords=preserve_coords,
      preload_b=bool(int(os.getenv("PRELOAD_B", "0"))),
      preload_b_safe_coords=bool(int(os.getenv("PRELOAD_B_SAFE_COORDS", "0"))),
      high_inputs=high_inputs,
      high_store=high_store,
      copy_b_probe=bool(int(os.getenv("COPY_B_PROBE", "0"))),
      thread_store=bool(int(os.getenv("THREAD_STORE", "0"))),
      repeat_first_store=bool(int(os.getenv("REPEAT_FIRST_STORE", "0"))),
      repair_row1_store=bool(int(os.getenv("REPAIR_ROW1_STORE", "0"))),
      repeat_each_store=bool(int(os.getenv("REPEAT_EACH_STORE", "0"))),
      post_constant=bool(int(os.getenv("POST", "0"))),
      stable_bx=fast and not preserve_coords, stable_ay=fast, low_a_coords=low_a,
      inc_coords=inc, persistent_coords=persistent,
      serial_b_cols=bool(int(os.getenv("SERIAL", "0"))),
      single_cols_all=bool(int(os.getenv("SINGLE_COLS_ALL", "0"))),
      first_sync_only=bool(int(os.getenv("FIRST_SYNC_ONLY", str(int(fast))))),
      no_store=bool(int(os.getenv("NO_STORE", "0"))),
      skip_a_loads=bool(int(os.getenv("SKIP_A_LOADS", "0"))),
      skip_b_loads=bool(int(os.getenv("SKIP_B_LOADS", "0"))),
      k_unroll=unroll, b_first=fast and ncols == 4 and not preserve_coords,
      k_count=None if dynamic_splits else k_count,
      coord_delay=int(os.getenv("COORD_DELAY", "-1" if fast else "4")),
      stable_settle_delay=int(os.getenv("STABLE_SETTLE_DELAY", "5")),
      row_sync=bool(int(os.getenv("ROW_SYNC", "0"))), store_row_shift=int(os.getenv("STORE_ROW_SHIFT", "10")),
      store_gap=int(os.getenv("STORE_GAP", "-1")),
      safe_b_y=bool(int(os.getenv("SAFE_B_Y", "0"))), sync_b_y=bool(int(os.getenv("SYNC_B_Y", "0"))),
      separate_b_coords=bool(int(os.getenv("SEPARATE_B_COORDS", "0"))),
      high_b_coords=bool(int(os.getenv("HIGH_B_COORDS", "0"))),
      reuse_separate_b_y=bool(int(os.getenv("REUSE_SEPARATE_B_Y", "0"))),
      persistent_b_coords=bool(int(os.getenv("PERSISTENT_B_COORDS", "0"))),
      interleave_second_pair=bool(int(os.getenv("INTERLEAVE_SECOND_PAIR", "0"))),
      pipeline=bool(int(os.getenv("PIPELINE", "0"))),
      acc_hr=int(os.getenv("ACC_HR")) if os.getenv("ACC_HR") else None,
      high_a_only=bool(int(os.getenv("HIGH_A_ONLY", "0"))),
      save_output_coords=bool(int(os.getenv("SAVE_OUTPUT_COORDS", "0"))),
      vector_init=bool(int(os.getenv("VECTOR_INIT", "0"))),
      dynamic_split_k=dynamic_splits,
      alu_order=os.getenv("ALU_ORDER", "auto"),
      first_cols_only=bool(int(os.getenv("FIRST_COLS_ONLY", "0"))), first_cols_offset=int(os.getenv("FIRST_COLS_OFFSET", "0")))
    merged_opt = os.getenv("MERGEDREGS")
    mergedregs = None if merged_opt is None else bool(int(merged_opt))
    native = bool(int(os.getenv("NATIVE_STORE", "0")))
    save_output = bool(int(os.getenv("SAVE_OUTPUT_COORDS", "0")))
    persistent_b = bool(int(os.getenv("PERSISTENT_B_COORDS", "0")))
    default_fregs = (23 if isolated else 30 if high_store else 28 if native else 19 if save_output and persistent_b else
                     18 if bool(int(os.getenv("HIGH_B_COORDS", "0"))) else
                     16 if bool(int(os.getenv("SAFE_B_Y", "0"))) else 11 if save_output or bool(int(os.getenv("THREAD_STORE", "0"))) else
                     8 if high_inputs and low_a else 10)
    acc_hr = int(os.getenv("ACC_HR", "0"))
    default_hregs = (28 if isolated else max(acc_hr + 4*ncols, 36 if bool(int(os.getenv("HIGH_A_ONLY", "0"))) else
                     44 if high_inputs and low_a else 48 if high_inputs else 32 if not compact else 12 + 4*ncols))
    lib = inject(env, io, sz, ro, shader, fregs=int(os.getenv("FREGS", str(default_fregs))),
                 hregs=int(os.getenv("HREGS", str(default_hregs))), mergedregs=mergedregs)
  if int(os.getenv("PRINT_META", "0")):
    asm = disasm(shader)
    print(f"shader_instrs={len(shader)//8} mad_f16={asm.count('mad.f16')} isam={asm.count('isam')} sy={asm.count('(sy)')}")
  if int(os.getenv("DUMP", "0")):
    print(disasm(shader))
    return
  ab, bb = upload(a), upload(b.reshape(k, n//4, 4))
  cb = Buffer("QCOM", max(1, dynamic_splits)*m*stride, dtypes.float if output_float else dtypes.half).allocate()
  cb.copyin(memoryview(np.zeros((max(1, dynamic_splits)*m, stride), np.float32 if output_float else np.float16)).cast("B"))
  specs = ([((0, dtypes.half, (m, stride//4, 4)),), ((0, dtypes.half, (m, k//4, 4)),),
            ((1, dtypes.half, (k, n//4, 4)),)] if image_store and not output_float else
           [((0, dtypes.float, (m, stride//4, 4)),), ((0, dtypes.half, (m, k//4, 4)),),
            ((1, dtypes.half, (k, n//4, 4)),)] if image_store else
           [((0, dtypes.half, (m, k//4, 4)),), ((1, dtypes.half, (k, n//4, 4)),),
            ((2, dtypes.half, (max(1, dynamic_splits)*m*stride,)),)])
  prg = dev.runtime("gemm_h", lib, buf_dtypes=specs)
  call_bufs = (cb._buf, ab._buf, bb._buf) if image_store else (ab._buf, bb._buf, cb._buf)
  tile_m = (threads//32)*4
  # Each workgroup covers 32 lanes * ncols half4 vectors = 128*ncols scalar columns;
  # thread count changes only the number of 4-row subtiles in Y.
  times = [prg(*call_bufs, global_size=(n//(128*ncols), (m//tile_m)*max(1, dynamic_splits), 1), local_size=(threads, 1, 1), wait=True)*1e3
           for _ in range(10)]
  if int(os.getenv("NO_STORE", "0")):
    print(f"K={k} ncols={ncols} compute_only_ms={min(times):.4f}")
    return
  got = np.empty((max(1, dynamic_splits)*m, stride), np.float32 if output_float else np.float16)
  cb.copyout(memoryview(got).cast("B"))
  if dynamic_splits:
    split_got = got.reshape(dynamic_splits, m, stride).astype(np.float32)
    if int(os.getenv("SPLIT_STATS", "0")):
      chunk = k//dynamic_splits
      split_expected = np.stack([a[:, s*chunk:(s+1)*chunk].astype(np.float32) @
                                 b[s*chunk:(s+1)*chunk].astype(np.float32) for s in range(dynamic_splits)])
      print("split_err", [[float(np.abs(split_got[x, :, :n]-split_expected[y]).mean())
                           for y in range(dynamic_splits)] for x in range(dynamic_splits)])
      print("split_norm", [float(np.abs(split_got[x, :, :n]).mean()) for x in range(dynamic_splits)])
    got = split_got.sum(axis=0)
  if int(os.getenv("RAW_STATS", "0")):
    nz = np.flatnonzero(got.reshape(-1))
    print("c_va", hex(cb._buf.va_addr), "raw_nonzero", len(nz), "head", nz[:128].tolist(), "tail", nz[-32:].tolist())
  if int(os.getenv("THREAD_STORE", "0")):
    raw, got = got.reshape(-1, 4, ncols, 4), np.empty_like(got)
    nz = np.flatnonzero(raw.reshape(-1))
    print("thread_nonzero_head", nz[:64].tolist(), "threads", np.unique(nz//(16*ncols))[:64].tolist(),
          "thread_count", len(np.unique(nz//(16*ncols))))
    # The thread-major kernel reserves tile slots using the physical output
    # stride, even when only a logical prefix of columns is launched.
    storage_gx_count = stride//(128*ncols)
    launched_gx_count = n//(128*ncols)
    for gy in range(m//16):
      for gx in range(launched_gx_count):
        for lid in range(128):
          tm, tid = lid//32, lid%32
          thread = (gy*storage_gx_count+gx)*128+lid
          col_base = gx*32*ncols+tid
          for row in range(4):
            for col in range(ncols): got[gy*16+tm*4+row, (col_base+col*32)*4:(col_base+col*32+1)*4] = raw[thread, row, col]
  got = got[:, :n]
  expected = (np.full((m, n), 1024.0, np.float32) if int(os.getenv("POST", "0")) else
              np.broadcast_to(b[0].astype(np.float32), (m, n)) if int(os.getenv("COPY_B_PROBE", "0")) else
              np.full((m, n), float(k), np.float32) if int(os.getenv("SKIP_A_LOADS", "0")) and int(os.getenv("SKIP_B_LOADS", "0")) else
              a[:, :k_count*4].astype(np.float32) @ b[:k_count*4].astype(np.float32))
  delta = np.abs(got.astype(np.float32)-expected)
  checked = np.ones((m, n), dtype=bool)
  if int(os.getenv("FIRST_COLS_ONLY", "0")):
    selected_parity = int(os.getenv("FIRST_COLS_OFFSET", "0")) & 1
    for block in range(n//128):
      if (block % ncols) % 2 != selected_parity:
        delta[:, block*128:(block+1)*128] = 0
        checked[:, block*128:(block+1)*128] = False
  correct = np.allclose(got[checked], expected[checked], rtol=2e-2, atol=2e-2)
  print(f"K={k} fast={fast} ms={min(times):.4f} max={delta.max():.8g} mean={delta.mean():.8g} "
        f"finite={np.isfinite(got[checked]).all()} allclose={correct}")
  print("samples", got[0, :16].tolist(), expected[0, :16].tolist())
  if pattern:
    print("pattern_blocks", [(x, got[0, x:x+8].tolist()) for x in range(0, n, 32)])
  print("worst", np.unravel_index(int(np.nanargmax(delta)), delta.shape),
        "col_means", [float(delta[:, x:x+128].mean()) for x in range(0, n, 128)],
        "row_means", [float(delta[x:x+16].mean()) for x in range(0, m, 16)])
  for out_block in (1, 3):
    x = out_block * 128
    print("block_match", out_block,
          [float(np.abs(got[:, x:x+128].astype(np.float32)-expected[:, y:y+128]).mean())
           for y in range(0, n, 128)])
  bad = np.argwhere(delta > 0.02)
  print("bad_count", len(bad), "bad_head", bad[:32].tolist())
  print("bad_rows", [(int(r), int((bad[:, 0] == r).sum())) for r in np.unique(bad[:, 0])],
        "bad_col_range", (int(bad[:, 1].min()), int(bad[:, 1].max())) if len(bad) else None)
  print("row1_match", [float(np.abs(got[1].astype(np.float32)-expected[r]).mean()) for r in range(16)])
  for probe_row in (127, 128, m-1):
    if probe_row >= m: continue
    probe_cols = checked[probe_row]
    row_delta = np.abs(expected[:, probe_cols] - got[probe_row, probe_cols].astype(np.float32)).mean(axis=1)
    nearest = np.argsort(row_delta)[:4]
    print("row_match", probe_row, [(int(r), float(row_delta[r])) for r in nearest])
  if not correct: raise SystemExit(1)


if __name__ == "__main__": main()
