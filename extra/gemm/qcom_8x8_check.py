#!/usr/bin/env python3
"""Random-matrix oracle for the high-throughput 8x8 IR3 GEMM."""
import os, hashlib
import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm import qcom_8x4_gemm as q8
from extra.gemm import qcom_intensity_gemm as q4
from extra.gemm.ir3asm import disasm, get_envelope, inject


def main():
  m, n, k = int(os.getenv("M", "128")), int(os.getenv("N", "512")), int(os.getenv("K", "192"))
  batch = int(os.getenv("BATCH", "1"))
  threads = int(os.getenv("THREADS", "128"))
  stride = int(os.getenv("STRIDE", str(max(1024, n))))
  k_start, k_count = int(os.getenv("K_START", "0")), int(os.getenv("K_COUNT", str(k//4)))
  rng = np.random.default_rng(int(os.getenv("SEED", "0")))
  a_np = (rng.standard_normal((batch*m, k))*0.05).astype(np.float16)
  b_np = (rng.standard_normal((batch*k, n))*0.05).astype(np.float16)
  batch_horizontal = batch > 1 and bool(int(os.getenv("BATCH_HORIZONTAL", "1")))
  batch_repeat_b = batch > 1 and not batch_horizontal and bool(int(os.getenv("BATCH_REPEAT_B", "0")))
  batch_repeat_b_x = batch > 1 and not batch_horizontal and bool(int(os.getenv("BATCH_REPEAT_B_X", "0")))
  b_storage = (np.concatenate([b_np[x*k:(x+1)*k] for x in range(batch) for _ in range(m//8)], axis=1)
               if batch_repeat_b_x else
               np.concatenate([b_np[x*k:(x+1)*k] for x in range(batch)], axis=1) if batch_horizontal else
               np.concatenate([b_np[x*k:(x+1)*k] for x in range(batch) for _ in range(m//8)])
               if batch_repeat_b else b_np)
  pattern = os.getenv("PATTERN", "")
  if pattern:
    a_np.fill(0)
    b_np.fill(0)
    if pattern == "row":
      a_np[:, 0] = np.arange(1, m+1, dtype=np.float16)
      b_np[0, :] = 1
    elif pattern == "col":
      a_np[:, 0] = 1
      b_np[0, :] = (np.arange(n, dtype=np.float16) % 251) + 1
    elif pattern.startswith("k"):
      kk = int(pattern[1:])
      a_np[:, kk] = np.arange(1, m+1, dtype=np.float16)
      b_np[kk, :] = 1
    elif pattern.startswith("cross"):
      ak, bk = map(int, pattern[5:].split("_"))
      a_np[:, ak] = np.arange(1, m+1, dtype=np.float16)
      b_np[bk, :] = 1
    elif pattern == "ones":
      a_np.fill(1)
      b_np.fill(1)
    else: raise ValueError(f"unknown PATTERN={pattern!r}")
  q8.M, q8.N, q8.K, q8.K4 = batch*m, stride, k, k//4
  dev = Device["QCOM"]
  image_store = bool(int(os.getenv("IMAGE_STORE", "0")))
  batch_const_mask = batch > 1 and bool(int(os.getenv("BATCH_CONST_MASK", "0")))
  batch_z = batch > 1 and bool(int(os.getenv("BATCH_Z", "0")))
  loop_instrs = -1
  if bool(int(os.getenv("COMPILER", "0"))):
    lib, _, _, _ = get_envelope(dev, q8.make_donor_src8(2, 128))
  else:
    wide = bool(int(os.getenv("WIDE", "0")))
    tri = bool(int(os.getenv("TRI", "0")))
    env_src = q4.make_direct_image_donor_src(4, threads) if image_store else q8.make_donor_src8(4, threads)
    if batch_const_mask:
      if not image_store: raise ValueError("BATCH_CONST_MASK currently requires IMAGE_STORE=1")
      groups_per_batch = m // ((threads//32)*8)
      env_src = env_src.replace("for(int k4=0", f"int batch=get_group_id(1)/{groups_per_batch};for(int k4=0")
      env_src = env_src.replace("k4*4+", f"batch*{k}+k4*4+")
    env, io, sz, ro = get_envelope(dev, env_src)
    if int(os.getenv("PERSISTENT8", "0")):
      shader, hregs, fregs, loop_instrs = q8.build_8x8_persistent_shader(dev, threads, k_count=k_count,
        store_row_shift=int(os.getenv("STORE_ROW_SHIFT", "10")), pipeline_b=bool(int(os.getenv("PERSISTENT_PIPELINE", "0"))),
        b_reuse_gap=int(os.getenv("B_REUSE_GAP", "0")), double_b=bool(int(os.getenv("PERSISTENT_DOUBLE_B", "0"))),
        rotate_b=bool(int(os.getenv("PERSISTENT_ROTATE_B", "0"))), pipeline_a=bool(int(os.getenv("PERSISTENT_PIPELINE_A", "0"))),
        one_sync=bool(int(os.getenv("PERSISTENT_ONE_SYNC", "0"))), one_sync_wait=int(os.getenv("PERSISTENT_ONE_SYNC_WAIT", "0")),
        stagger_b=bool(int(os.getenv("PERSISTENT_STAGGER_B", "0"))),
        stagger_rows=int(os.getenv("STAGGER_ROWS", "2")), masked_prefetch_a4=bool(int(os.getenv("MASKED_PREFETCH_A4", "0"))),
        lagged_a4=bool(int(os.getenv("LAGGED_A4", "0"))), dual_a_tile=bool(int(os.getenv("DUAL_A_TILE", "0"))),
        stream_a4_gap=int(os.getenv("STREAM_A4_GAP", "-1")),
        dynamic_a4_dual=bool(int(os.getenv("DYNAMIC_A4_DUAL", "0"))),
        dynamic_a4_wait=int(os.getenv("DYNAMIC_A4_WAIT", "0")),
        dynamic_b_prefetch=bool(int(os.getenv("DYNAMIC_B_PREFETCH", "0"))),
        dynamic_b_rows=int(os.getenv("DYNAMIC_B_ROWS", "1")),
        dynamic_b_gap=int(os.getenv("DYNAMIC_B_GAP", "0")),
        rotate_low_banks=bool(int(os.getenv("ROTATE_LOW_BANKS", "0"))),
        rotate_no_prefetch=bool(int(os.getenv("ROTATE_NO_PREFETCH", "0"))),
        batch_m=m if batch > 1 and bool(int(os.getenv("BATCH_SHADER", "1"))) else 0,
        batch_n=n if batch > 1 and bool(int(os.getenv("BATCH_SHADER", "1"))) else 0,
        batch_k=k if batch > 1 and bool(int(os.getenv("BATCH_SHADER", "1"))) else 0,
        batch_b_offset=bool(int(os.getenv("BATCH_B_OFFSET", "1"))),
        batch_row_offset=bool(int(os.getenv("BATCH_ROW_OFFSET", "1"))),
        batch_horizontal=batch_horizontal,
        batch_repeat_b=batch_repeat_b,
        batch_repeat_b_x=batch_repeat_b_x,
        batch_fixed_b=-2 if batch_z else int(os.getenv("BATCH_FIXED_B", "-1")),
        batch_const_mask=batch_const_mask,
        image_store_gap=int(os.getenv("IMAGE_STORE_GAP", "16")),
        image_store=image_store)
    elif int(os.getenv("PACKED_B8", "0")):
      shader, hregs, fregs, loop_instrs = q8.build_8x8_bpacked_shader(dev, threads, coord_delay=int(os.getenv("ADELAY", "5")),
        merged_alias=bool(int(os.getenv("PACKED_B_ALIAS", "0"))))
    elif int(os.getenv("PACKED8", "0")):
      shader, hregs, fregs, loop_instrs = q8.build_8x8_packed8_shader(dev, threads, coord_delay=int(os.getenv("ADELAY", "2")))
    elif tri:
      shader, hregs, fregs, loop_instrs = q8.build_8x4_shader(dev, 128, os.getenv("TRI_VARIANT", "serial"), 3,
        a_coord_delay=int(os.getenv("ADELAY", "-1")), b_coord_delay=int(os.getenv("BDELAY", "-1")),
        post_constant=bool(int(os.getenv("POST_CONSTANT", "0"))), image_store=image_store)
    elif wide:
      if image_store: raise ValueError("WIDE image store is not implemented")
      shader, hregs, fregs, loop_instrs = q8.build_8x16_split_a_unroll_shader(dev, 128, k_unroll=int(os.getenv("KUNROLL", "4")),
        b_coord_delay=int(os.getenv("BDELAY", "0")), fast_coords=True, safe_coords=bool(int(os.getenv("SAFE_COORDS", "1"))),
        add256_store_mode=os.getenv("STORE_MODE", "tight"), alu_order=os.getenv("ALU_ORDER", "row_col_kk"),
        post_constant=bool(int(os.getenv("POST_CONSTANT", "0"))),
        skip_a_loads=bool(int(os.getenv("SKIP_A_LOADS", "0"))), skip_b_loads=bool(int(os.getenv("SKIP_B_LOADS", "0"))),
        store_row_shift=int(os.getenv("STORE_ROW_SHIFT", "10")))
    elif int(os.getenv("LIFETIME", "0")):
      shader, hregs, fregs, loop_instrs = q8.build_8x8_lifetime_shader(dev, 128, k_unroll=int(os.getenv("KUNROLL", "4")),
        b_coord_delay=int(os.getenv("BDELAY", "0")), a_coord_delay=int(os.getenv("ADELAY", "0")),
        k_start=k_start, k_count=k_count, post_sequence=bool(int(os.getenv("POST_SEQUENCE", "0"))))
    elif int(os.getenv("SELF_COORDS", "0")):
      shader, hregs, fregs, loop_instrs = q8.build_8x8_selfcoord_shader(
        dev, 128, coord_delay=int(os.getenv("ADELAY", "0")), post_sequence=bool(int(os.getenv("POST_SEQUENCE", "0"))))
    elif int(os.getenv("BASE", "0")):
      shader, hregs, fregs, loop_instrs = q8.build_8x8_split_a_shader(dev, 128,
        a_coord_delay=int(os.getenv("ADELAY", "4")), b_coord_delay=int(os.getenv("BDELAY", "4")),
        post_constant=bool(int(os.getenv("POST_CONSTANT", "0"))),
        thread_store_gx=n//256 if int(os.getenv("THREAD_STORE", "0")) else 0,
        thread_store_lid_reg=None if os.getenv("SAVE_REG", "r28.x") == "none" else os.getenv("SAVE_REG", "r28.x"),
        thread_store_group_regs=("r36.y", "r36.z") if int(os.getenv("SAVE_GROUPS", "0")) else None,
        row_sync=bool(int(os.getenv("ROW_SYNC", "0"))), reserved_out=int(os.getenv("RESERVED_OUT", "-1")))
    else:
      hist = bool(int(os.getenv("HIST", "0")))
      common = dict(k_unroll=int(os.getenv("KUNROLL", "8")), b_coord_delay=int(os.getenv("BDELAY", "0")),
        fast_coords=True, prefetch_next_b=bool(int(os.getenv("PREFETCH", "0"))), add256_store_mode=os.getenv("STORE_MODE", "tight"),
        prefetch_next_a=bool(int(os.getenv("PREFETCH_A", "0"))),
        grouped_b=bool(int(os.getenv("GROUPED_B", "0"))), grouped_b_cols=bool(int(os.getenv("GROUPED_B_COLS", "0"))),
        stream_col1=bool(int(os.getenv("STREAM_COL1", "0"))), stream_col1_sync=bool(int(os.getenv("STREAM_COL1_SYNC", "0"))),
        add256_gap=int(os.getenv("ADD256_GAP", "16")),
        add256_offset_before_gap=bool(int(os.getenv("ADD256_OFFSET_BEFORE_GAP", "0"))),
        alu_order=os.getenv("ALU_ORDER", "row_col_kk"),
        post_constant=bool(int(os.getenv("POST_CONSTANT", "0"))))
      if hist:
        shader, hregs, fregs, loop_instrs = q8.build_8x8_split_a_unroll_shader(dev, 128, **common)
      else:
        shader, hregs, fregs, loop_instrs = q8.build_8x8_split_a_unroll_shader(dev, threads, **common, k_start=k_start, k_count=k_count,
        thread_store_gx=n//256 if int(os.getenv("THREAD_STORE", "0")) else 0,
        post_sequence=bool(int(os.getenv("POST_SEQUENCE", "0"))),
        a_coord_delay=int(os.getenv("ADELAY", "4")), unroll_gap=int(os.getenv("GAP", "0")),
        relaxed_sync=bool(int(os.getenv("RELAXED_SYNC", "0"))), sync_mask=int(os.getenv("SYNC_MASK", "7"), 0),
        sync_wait=int(os.getenv("SYNC_WAIT", "0")), high_inputs=bool(int(os.getenv("HIGH_INPUTS", "0"))), image_store=image_store,
        mid_acc=bool(int(os.getenv("MID_ACC", "0"))),
        safe_coords=bool(int(os.getenv("SAFE_COORDS", "0"))), low_stable_coords=bool(int(os.getenv("LOW_STABLE_COORDS", "0"))),
        triple_coords=bool(int(os.getenv("TRIPLE_COORDS", "0"))),
        dual_a_coords=bool(int(os.getenv("DUAL_A_COORDS", "0"))),
        high_pair_coords=bool(int(os.getenv("HIGH_PAIR_COORDS", "0"))),
        high_a=bool(int(os.getenv("HIGH_A", "0"))),
        low_a=bool(int(os.getenv("LOW_A", "0"))),
        high_pair_b=bool(int(os.getenv("HIGH_PAIR_B", "0"))), high_pair_a=bool(int(os.getenv("HIGH_PAIR_A", "0"))),
        serial_safe_coords=bool(int(os.getenv("SERIAL_SAFE_COORDS", "0"))),
        separate_coords=bool(int(os.getenv("SEPARATE_COORDS", "0"))), buffer_a=bool(int(os.getenv("BUFFER_A", "0"))),
        prefetch_loop_b=bool(int(os.getenv("PREFETCH_LOOP_B", "0"))), preload_a8=bool(int(os.getenv("PRELOAD_A8", "0"))),
        reuse_b=bool(int(os.getenv("REUSE_B", "0"))), row_stream=bool(int(os.getenv("ROW_STREAM", "0"))),
        phase_stream=bool(int(os.getenv("PHASE_STREAM", "0"))), split_low_pairs=bool(int(os.getenv("SPLIT_LOW_PAIRS", "0"))),
        quad_a=bool(int(os.getenv("QUAD_A", "0"))), quad_map=os.getenv("QUAD_MAP", "0123"),
        sampler_source_sync=bool(int(os.getenv("SOURCE_SYNC", "0"))),
        stream_b_a8=bool(int(os.getenv("STREAM_B_A8", "0"))), store_row_shift=int(os.getenv("STORE_ROW_SHIFT", "10")),
        source_hold_delay=int(os.getenv("SOURCE_HOLD_DELAY", "-1")), one_sync_tile=bool(int(os.getenv("ONE_SYNC_TILE", "0"))),
        interleave_a4=bool(int(os.getenv("INTERLEAVE_A4", "0"))), interleave_a_reuse_gap=int(os.getenv("A_REUSE_GAP", "0")),
        single_high_coord=bool(int(os.getenv("SINGLE_HIGH_COORD", "0"))))
    assert len(shader) <= sz
    if int(os.getenv("DISASM", "0")): print(disasm(shader))
    lib = inject(env, io, sz, ro, shader, fregs=int(os.getenv("FREGS", str(fregs))), hregs=int(os.getenv("HREGS", str(hregs))),
                 mergedregs=False if bool(int(os.getenv("SEPARATE_REGS", "0"))) else None)
    if int(os.getenv("PRINT_META", "0")): print("shader_meta", fregs, hregs, len(shader), loop_instrs, hashlib.sha1(lib).hexdigest()[:8])
  a, b = Buffer("QCOM", a_np.size, dtypes.half).allocate(), Buffer("QCOM", b_storage.size, dtypes.half).allocate()
  c = Buffer("QCOM", batch*m*stride, dtypes.half).allocate()
  q8.buf_copyin(a, memoryview(a_np).cast("B"))
  q8.buf_copyin(b, memoryview(b_storage).cast("B"))
  if not int(os.getenv("NO_INIT", "0")):
    q8.buf_copyin(c, memoryview(np.zeros(batch*m*stride, dtype=np.float16)).cast("B"))
  packed8 = bool(int(os.getenv("PACKED8", "0")))
  packed_b8 = bool(int(os.getenv("PACKED_B8", "0")))
  specs = ([((0, dtypes.half, (batch*m, stride//4, 4)),), ((0, dtypes.half, (batch*m, k//4, 4)),),
            ((1, dtypes.half, (k, batch*(m//8)*n//4, 4)),) if batch_repeat_b_x else
            ((1, dtypes.half, (k, batch*n//4, 4)),) if batch_horizontal else
            ((1, dtypes.half, ((batch*k*(m//8) if batch_repeat_b else batch*k), n//4, 4)),)] if image_store else
           [((0, dtypes.uint32, (m, k//8, 4)),), ((0, dtypes.uint32, (k, n//8, 4)),), ((0, dtypes.half, None),)] if packed8 else
           [((0, dtypes.half, (m, k//4, 4)),), ((0, dtypes.uint32, (k, n//8, 4)),), ((0, dtypes.half, None),)] if packed_b8 else
           [((0, dtypes.half, (batch*m, k//4, 4)),),
            ((0, dtypes.half, (k, batch*n//4, 4)),) if batch_horizontal else ((0, dtypes.half, (batch*k, n//4, 4)),),
            ((0, dtypes.half, None),)])
  prg = dev.runtime("gemm_h", lib, buf_dtypes=specs)
  if int(os.getenv("PRINT_META", "0")):
    print("buffer_specs", specs)
    print("runtime_meta", {k:v for k,v in vars(prg).items() if k in ("wgid", "lid", "max_threads", "prg")})
  call_bufs = (c._buf, a._buf, b._buf) if image_store else (a._buf, b._buf, c._buf)
  tile_n = 384 if bool(int(os.getenv("TRI", "0"))) else 512 if bool(int(os.getenv("WIDE", "0"))) else 256
  tile_m = (threads//32)*8
  global_size = ((n//tile_n, m//tile_m, batch) if batch_z else
                 (batch*n//tile_n, m//tile_m, 1) if batch_horizontal else
                 (n//tile_n, batch*m//tile_m, 1))
  times = [prg(*call_bufs, global_size=global_size,
               local_size=(threads, 1, 1), wait=True) for _ in range(int(os.getenv("BENCH_RUNS", "10")))]
  elapsed = min(times)
  got = np.empty(batch*m*stride, dtype=np.float16)
  q8.buf_copyout(c, memoryview(got).cast("B"))
  if int(os.getenv("RAW_STATS", "0")):
    nz = np.flatnonzero(got)
    print("raw_nonzero", nz.size, "first", nz[:64].tolist(), "last", nz[-64:].tolist(),
          "values", np.unique(got[nz])[:16].tolist())
  if int(os.getenv("THREAD_STORE", "0")) or int(os.getenv("DECODE_THREAD", "0")):
    raw, matrix = got[:m*n].reshape(-1, 8, 2, 4), np.empty((m, n), np.float16)
    if pattern:
      print("raw_lids=", [[float(raw[lid, row, 0, 0]) for row in range(8)] for lid in range(0, 128, 8)])
    gx_count = n//256
    for gy in range(m//tile_m):
      for gx in range(gx_count):
        for lid in range(threads):
          tm, tid = lid//32, lid%32
          thread = (gy*gx_count+gx)*threads+lid
          for row in range(8):
            for col in range(2):
              x = (gx*64+tid+col*32)*4
              matrix[gy*tile_m+tm*8+row, x:x+4] = raw[thread, row, col]
    got = matrix.astype(np.float32)
  else: got = got.reshape(batch*m, stride)[:, :n].astype(np.float32)
  if int(os.getenv("POST_SEQUENCE", "0")):
    tile = np.empty((8, 256), np.float32)
    for row in range(8):
      for col in range(2): tile[row, col*128:(col+1)*128] = row*2+col+1
    expected = np.tile(tile, (m//8, n//256))
  else: expected = (np.full((batch*m, n), 1024, np.float32) if int(os.getenv("POST_CONSTANT", "0")) else
              np.concatenate([a_np[x*m:(x+1)*m, k_start*4:(k_start+k_count)*4].astype(np.float32) @
                              b_np[x*k+k_start*4:x*k+(k_start+k_count)*4].astype(np.float32)
                              for x in range(batch)]))
  delta = np.abs(expected-got)
  if (reserved_out := int(os.getenv("RESERVED_OUT", "-1"))) >= 0:
    row, col = divmod(reserved_out, 2)
    delta[row::8, col*128:(col+1)*128] = 0
  correct = np.allclose(expected, got, rtol=2e-2, atol=2e-2)
  gflops = batch*2*m*n*(k_count*4)/elapsed/1e9
  print(f"shape={batch}x{m}x{n}x{k_count*4} accumulate=fp16 elapsed_ms={elapsed*1e3:.3f} gflops={gflops:.1f} "
        f"max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} allclose={correct}")
  bad = ~np.isfinite(got) | (delta > .02)
  bad_idx = np.argwhere(bad)
  print(f"bad_count={bad_idx.shape[0]}")
  if int(os.getenv("VERBOSE", "0")):
    for r in range(8): print(f"row{r} expected={expected[r,:8].tolist()} got={got[r,:8].tolist()}")
    print("block_max=", [[float(delta[r:r+8, c:c+128].max()) for c in range(0, n, 128)] for r in range(0, m, 8)])
    print("local_rows=", [(lr, float(delta[lr::8].max()), float(delta[lr::8].mean())) for lr in range(8)])
    print("bad_by_row=", [(int(r), int(bad[r].sum())) for r in np.flatnonzero(bad.any(axis=1))])
    print("bad_first=", [(int(r), int(c), float(expected[r, c]), float(got[r, c])) for r, c in bad_idx[:64]])
  if int(os.getenv("POST_SEQUENCE", "0")):
    print("sequence_blocks=", [[np.unique(got[row, col:col+128], return_counts=True) for col in range(0, n, 128)] for row in range(8)])
  if int(os.getenv("VERBOSE", "0")):
    row0_matches = np.abs(expected-got[0]).mean(axis=1)
    print("row0_matches=", [(int(i), float(row0_matches[i])) for i in np.argsort(row0_matches)[:8]])
    if batch > 1:
      for row in range(0, batch*m, (threads//32)*8):
        candidates = [a_np[row].astype(np.float32) @ b_np[x*k:(x+1)*k].astype(np.float32) for x in range(batch)]
        print("batch_map=", row, [(x, float(np.abs(c-got[row]).mean())) for x, c in enumerate(candidates)])
  if int(os.getenv("VERBOSE", "0")) and not int(os.getenv("POST_SEQUENCE", "0")) and not int(os.getenv("POST_CONSTANT", "0")):
    contrib = np.stack([a_np[0, kk*4:kk*4+4].astype(np.float32) @
                        b_np[kk*4:kk*4+4].astype(np.float32) for kk in range(k_start, k_start+k_count)])
    excluded = np.abs((expected[0][None, :]-contrib)-got[0]).mean(axis=1)
    prefixes = np.abs(np.cumsum(contrib, axis=0)-got[0]).mean(axis=1)
    print("row0_k=", "exclude", [(k_start+int(i), float(excluded[i])) for i in np.argsort(excluded)[:4]],
          "prefix", [(k_start+int(i)+1, float(prefixes[i])) for i in np.argsort(prefixes)[:4]])
    if k_count == 1:
      cs = np.stack([a_np[:, k_start*4+j:k_start*4+j+1].astype(np.float32) @
                     b_np[k_start*4+j:k_start*4+j+1].astype(np.float32) for j in range(4)])
      subset = [(mask, float(np.abs(sum((cs[j] for j in range(4) if mask & (1<<j)), np.zeros_like(got))-got).mean())) for mask in range(16)]
      print("component_subsets=", sorted(subset, key=lambda x:x[1])[:8])
  if not correct: raise SystemExit(1)


if __name__ == "__main__": main()
