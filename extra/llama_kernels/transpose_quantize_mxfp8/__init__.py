from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import THREADS_PER_WG, alloc_like, dname_of, compile_hip, owned_empty

TILE_N = THREADS_PER_WG   # 256
BLK = 32

@functools.cache
def _custom_transpose_quantize_mxfp8(q:UOp, e8:UOp, *args:UOp, dname:str) -> UOp:
  assert len(args) in (1, 2, 5)
  if len(args) == 1: bias_partial, row_q, row_e8, row_si, g = None, None, None, None, args[0]
  elif len(args) == 2: bias_partial, row_q, row_e8, row_si, g = args[0], None, None, None, args[1]
  else: bias_partial, row_q, row_e8, row_si, g = args
  M, N = g.shape
  assert bias_partial is None or bias_partial.shape == (M // BLK, N)
  assert (row_q is None) == (row_e8 is None) == (row_si is None)
  gptoss_qkv_dual_tq = bool(getenv("GPTOSS_QKV_TQ_DUAL", 0)) and bias_partial is not None and row_q is not None and \
    (M, N) == (16384, 5120)
  col_si_alias = gptoss_qkv_dual_tq and e8.dtype == dtypes.uint32 and e8.shape == (M // (4 * BLK), N)
  assert col_si_alias or (e8.dtype == dtypes.uint8 and e8.shape == (N, M // BLK))
  # The exact GPT-OSS dgrad GEMM consumes row_q + packed row_si; gemm_mxfp8.cpp names its raw-e8 pointer
  # a_e8_unused and never reads it.  In that path the wrapper aliases this ABI-only slot to row_si, identified here
  # by the otherwise-invalid uint32 row_e8 shape, so the producer can omit both the dead write and allocation.
  row_e8_alias = gptoss_qkv_dual_tq and row_e8 is not None and row_si is not None and \
    row_e8.dtype == dtypes.uint32 and row_e8.shape == row_si.shape
  tq_threads = getenv("GPTOSS_QKV_TQ_DUAL_THREADS", 128) if gptoss_qkv_dual_tq else \
    getenv("GPTOSS_MOE_TQ_THREADS", getenv("TQ_THREADS", THREADS_PER_WG)) \
    if bias_partial is not None and row_q is None and M == 73728 and N in (3072, 5888) else getenv("TQ_THREADS", THREADS_PER_WG)
  assert tq_threads in (64, 128, 256) and N % tq_threads == 0
  assert row_q is None or tq_threads == THREADS_PER_WG or (gptoss_qkv_dual_tq and tq_threads == 128), \
    "dual row output requires a complete 128-column scale superblock"
  if row_q is not None:
    assert row_q.shape == (M, N) and row_si.shape == (N // (4 * BLK), M)
    assert row_e8_alias or (row_e8.dtype == dtypes.uint8 and row_e8.shape == (M, N // BLK))
  num_wg = (M // BLK) * (N // tq_threads)
  threads, workgroups = UOp.special(tq_threads, "lidx0"), UOp.special(num_wg, "gidx0")
  mem = M * N * 2 + M * N + (M // BLK) * N   # read bf16, write columnwise fp8 + e8
  if bias_partial is not None: mem += (M // BLK) * N * 4
  if row_q is not None: mem += M * N + 2 * M * (N // BLK)
  kargs = (q.base, e8.base)
  if bias_partial is not None: kargs += (bias_partial.base,)
  if row_q is not None: kargs += (row_q.base, row_e8.base, row_si.base)
  kargs += (g.base,)
  suffix = "_bias_dual" if row_q is not None else "_bias" if bias_partial is not None else ""
  gptoss_down_tq = bias_partial is not None and row_q is None and (M, N) == (73728, 3072)
  tq_direct_load = getenv("GPTOSS_DOWN_TQ_DIRECT_LOAD", 0) if gptoss_down_tq else \
    getenv("GPTOSS_QKV_TQ_DUAL_DIRECT_LOAD", 0) if gptoss_qkv_dual_tq else \
    getenv("GPTOSS_MOE_TQ_DIRECT_LOAD", 1) if bias_partial is not None and row_q is None and (M, N) == (73728, 5888) else 0
  if gptoss_down_tq or gptoss_qkv_dual_tq: suffix += f"_dl{int(bool(tq_direct_load))}"
  row_si_byte_store = gptoss_qkv_dual_tq and tq_threads == 128 and bool(getenv("GPTOSS_QKV_TQ_ROW_SI_BYTE_STORE", 0))
  if row_si_byte_store: suffix += "_sib1"
  if row_e8_alias: suffix += "_sionly"
  if col_si_alias: suffix += "_csionly"
  outputs = (q, e8) + ((bias_partial,) if bias_partial is not None else ()) + \
            ((row_q, row_si) if row_e8_alias else (row_q, row_e8, row_si) if row_q is not None else ())
  zero = UOp.const(0, dtypes.int32)
  accesses = tuple(x.index(zero).store(UOp.const(0, x.dtype)) for x in outputs) + (g.index(zero).load(),)
  sink = UOp.sink(*kargs, *accesses, threads, workgroups,
                  arg=KernelInfo(f"transpose_quantize_mxfp8{suffix}_{M}_{N}" + (f"_t{tq_threads}" if tq_threads != THREADS_PER_WG else ""),
                                 estimates=Estimates(ops=M*N, mem=mem)))
  src = (pathlib.Path(__file__).parent/"transpose_quantize_mxfp8.cpp").read_text()
  defines = [f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DTHREADS_PER_WG={tq_threads}",
             f"-DNATIVE_MXFP8_CVT={getenv('NATIVE_MXFP8_CVT', 0)}", f"-DTQ_BIAS_PARTIAL={int(bias_partial is not None)}",
             f"-DTQ_ROW_OUTPUT={int(row_q is not None)}", f"-DTQ_DIRECT_FULL_LOAD={int(bool(tq_direct_load))}",
             f"-DTQ_ROW_SI_BYTE_STORE={int(row_si_byte_store)}", f"-DTQ_ROW_E8_STORE={int(not row_e8_alias)}",
             f"-DTQ_COL_SI_BYTE_STORE={int(col_si_alias)}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_transpose_quantize_mxfp8_padded(q:UOp, e8:UOp, *args:UOp, dname:str) -> UOp:
  assert len(args) in (1, 2, 5)
  if len(args) == 1: bias_partial, row_q, row_e8, row_si, g = None, None, None, None, args[0]
  elif len(args) == 2: bias_partial, row_q, row_e8, row_si, g = args[0], None, None, None, args[1]
  else: bias_partial, row_q, row_e8, row_si, g = args
  M, src_n = g.shape
  out_n, out_m = q.shape
  assert out_m == M and src_n <= out_n and e8.shape == (out_n, M // BLK)
  assert bias_partial is None or bias_partial.shape == (M // BLK, out_n)
  assert (row_q is None) == (row_e8 is None) == (row_si is None)
  # A single-wave tile is a better fit for the GPT-OSS LM-head operands: it keeps the per-wave transpose work
  # unchanged while reducing LDS and avoiding multi-wave workgroup residency constraints. Production ROCm 7.1
  # ABBA64 on GPUs 2/3 measured -15.1% for weight and -5.2/-5.7% for hidden, byte-exact for q/raw/packed scales.
  gptoss_lmhead_operand = bias_partial is None and row_q is None and \
    (M, src_n, out_n) in ((128256, 2880, 3072), (16384, 2880, 3072))
  tq_threads = getenv("GPTOSS_LMHEAD_TQ_THREADS", 64) if gptoss_lmhead_operand else getenv("TQ_THREADS", THREADS_PER_WG)
  assert tq_threads in (64, 128, 256) and out_n % tq_threads == 0
  assert row_q is None or tq_threads == THREADS_PER_WG, "dual row output requires the 256-column tile"
  if row_q is not None:
    assert row_q.shape == (M, out_n) and row_e8.shape == (M, out_n // BLK) and row_si.shape == (out_n // (4 * BLK), M)
  assert M % BLK == 0 and src_n % BLK == 0 and out_n % TILE_N == 0
  threads = UOp.special(tq_threads, "lidx0")
  workgroups = UOp.special((M // BLK) * (out_n // tq_threads), "gidx0")
  kargs = (q.base, e8.base)
  if bias_partial is not None: kargs += (bias_partial.base,)
  if row_q is not None: kargs += (row_q.base, row_e8.base, row_si.base)
  kargs += (g.base,)
  suffix = "_bias_dual" if row_q is not None else "_bias" if bias_partial is not None else ""
  mem = M * (src_n * 2 + out_n) + out_n * (M // BLK)
  if bias_partial is not None: mem += (M // BLK) * out_n * 4
  if row_q is not None: mem += M * out_n + 2 * M * (out_n // BLK)
  outputs = (q, e8) + ((bias_partial,) if bias_partial is not None else ()) + \
            ((row_q, row_e8, row_si) if row_q is not None else ())
  zero = UOp.const(0, dtypes.int32)
  accesses = tuple(x.index(zero).store(UOp.const(0, x.dtype)) for x in outputs) + (g.index(zero).load(),)
  sink = UOp.sink(*kargs, *accesses, threads, workgroups,
                  arg=KernelInfo(f"transpose_quantize_mxfp8_padded{suffix}_{M}_{src_n}_{out_n}" +
                                 (f"_t{tq_threads}" if tq_threads != THREADS_PER_WG else ""),
                                 estimates=Estimates(ops=M*out_n, mem=mem)))
  src = (pathlib.Path(__file__).parent/"transpose_quantize_mxfp8.cpp").read_text()
  defines = [f"-DM_DIM={M}", f"-DN_DIM={out_n}", f"-DSRC_N_DIM={src_n}", f"-DTHREADS_PER_WG={tq_threads}",
             f"-DNATIVE_MXFP8_CVT={getenv('NATIVE_MXFP8_CVT', 0)}", f"-DTQ_BIAS_PARTIAL={int(bias_partial is not None)}",
             f"-DTQ_ROW_OUTPUT={int(row_q is not None)}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_transpose_requantize_mxfp8(q:UOp, e8:UOp, si:UOp, *args:UOp, dname:str, down_dgrad:bool=False) -> UOp:
  assert len(args) in (2, 3)
  packed_si, src_q, src_e8 = (None, *args) if len(args) == 2 else args
  E, M, N = (1, *src_q.shape) if src_q.ndim == 2 else src_q.shape
  gptoss_fc1 = (E, M, N) == (32, 5888, 3072)
  gptoss_down = down_dgrad and (E, M, N) == (32, 3072, 3072)
  gptoss_qkv = bool(getenv("GPTOSS_QKV_REQUANT_PACKED", 0)) and packed_si is not None and (E, M, N) == (1, 16384, 3072)
  assert not down_dgrad or gptoss_down, f"down dgrad requant requires the exact GPT-OSS down weight, got {(E, M, N)}"
  # The FC1 dgrad consumes q+packed SI; its raw e8 argument is ABI-only.  That caller aliases this slot to the
  # source e8 tensor (different shape from the ordinary transposed output) so the kernel can omit a scattered,
  # otherwise dead e8 write and avoid allocating another 18 MiB output.
  si_only = (gptoss_fc1 and packed_si is not None and e8.shape == src_e8.shape) or gptoss_down or \
    (gptoss_qkv and bool(getenv("GPTOSS_QKV_REQUANT_SI_ONLY", 0)) and e8.shape == src_e8.shape)
  split_m32 = bool(getenv("GPTOSS_FC1_REQUANT_M32", 1)) if gptoss_fc1 else \
    bool(getenv("GPTOSS_DOWN_REQUANT_M32", 1)) if gptoss_down else \
    bool(getenv("GPTOSS_QKV_REQUANT_M32", 0)) if gptoss_qkv else bool(getenv("REQUANT_M32", 0))
  tq_threads = getenv("GPTOSS_FC1_REQUANT_THREADS", 128) if gptoss_fc1 else \
    getenv("GPTOSS_DOWN_REQUANT_THREADS", 128) if gptoss_down else \
    getenv("GPTOSS_QKV_REQUANT_THREADS", 128) if gptoss_qkv else THREADS_PER_WG
  assert tq_threads in (64, 128, 256) and N % tq_threads == 0
  # The exact dgrad consumer skips both zero-padded FC1 weight dimensions. Keep the physical output ABI while
  # launching only its live (N=2880, K=5760) rectangle. This callback has no other trimmed caller.
  trim_dgrad = int(bool(getenv("GPTOSS_REQUANT_TRIM_DGRAD", 1))) if gptoss_fc1 and si_only and split_m32 and tq_threads == 128 else 0
  native_mxfp8 = getenv("NATIVE_MXFP8_CVT", 0)
  direct_bf16 = int(bool(getenv("GPTOSS_FC1_REQUANT_DIRECT_BF16", 0))) if trim_dgrad and native_mxfp8 else \
    int(bool(getenv("GPTOSS_DOWN_REQUANT_DIRECT_BF16", 0))) if gptoss_down and packed_si is not None and si_only and \
      split_m32 and tq_threads == 128 and native_mxfp8 else 0
  bfe_scale = int(bool(getenv("GPTOSS_FC1_REQUANT_BFE_SCALE", 1))) if gptoss_fc1 and trim_dgrad and direct_bf16 else \
    int(bool(getenv("GPTOSS_DOWN_REQUANT_BFE_SCALE", 0))) if gptoss_down and direct_bf16 else 0
  paired_down = gptoss_down and direct_bf16 and bfe_scale and bool(getenv("GPTOSS_DOWN_REQUANT_PAIR", 0))
  # Compact only the 5760-wide contraction grid. Retaining 24 physical N slots keeps the fast tile decode/order;
  # padded N lanes return before loads. Compacting both dimensions measured slower despite launching less work.
  launch_m, launch_n = (5760, N) if trim_dgrad else (M, N)
  num_wg = E * (launch_m // (BLK if split_m32 else 4 * BLK)) * ((launch_n + tq_threads - 1) // tq_threads)
  # Paired lanes keep the 128-column tile and its workgroup count, but need only one 64-lane wave.
  launch_threads = 64 if paired_down else tq_threads
  threads, workgroups = UOp.special(launch_threads, "lidx0"), UOp.special(num_wg, "gidx0")
  elems = E * launch_m * launch_n
  mem = 2 * elems + 3 * (elems // BLK)
  sink = UOp.sink(q.base, e8.base, si.base, *((packed_si.base,) if packed_si is not None else ()), src_q.base, src_e8.base,
                  threads, workgroups,
                  arg=KernelInfo(f"transpose_requantize_mxfp8_{E}_{M}_{N}" + ("_m32" if split_m32 else "") +
                                 ("_sionly" if si_only else "") +
                                 (f"_trim{trim_dgrad}" if trim_dgrad else "") +
                                 ("_dbf161" if direct_bf16 else "") +
                                 ("_bfe1" if bfe_scale else "") +
                                 ("_down" if gptoss_down else "") +
                                 ("_pair1" if paired_down else "") +
                                 (f"_t{launch_threads}" if launch_threads != THREADS_PER_WG else ""),
                                 estimates=Estimates(ops=2*elems, mem=mem)))
  src = (pathlib.Path(__file__).parent/"transpose_quantize_mxfp8.cpp").read_text()
  defines = [f"-DE_DIM={E}", f"-DM_DIM={M}", f"-DN_DIM={N}", f"-DTHREADS_PER_WG={tq_threads}",
             "-DTRANSPOSE_REQUANTIZE=1", f"-DNATIVE_MXFP8_CVT={native_mxfp8}",
             f"-DREQUANT_M32={int(split_m32)}", f"-DGPTOSS_REQUANT_PACKED_INPUT={int(packed_si is not None)}",
             f"-DGPTOSS_REQUANT_SI_ONLY={int(si_only)}", f"-DGPTOSS_REQUANT_TRIM_DGRAD={int(trim_dgrad)}",
             f"-DGPTOSS_REQUANT_DIRECT_BF16={direct_bf16}", f"-DGPTOSS_REQUANT_BFE_SCALE={bfe_scale}"]
  if paired_down: defines.append("-DGPTOSS_REQUANT_PAIR=1")
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def transpose_quantize_mxfp8(g:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  # fused g.T quantize: returns (q, e8, si) == quantize_mxfp8(g.T) — q (N,M) fp8, e8 (N, M/32), si packed (M/128, N)
  assert g.ndim == 2 and g.dtype == dtypes.bfloat16, f"{g.shape} {g.dtype}"
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE, mx_pack
  M, N = g.shape
  assert M % BLK == 0 and N % TILE_N == 0, f"M={M} must%{BLK}, N={N} must%{TILE_N}"
  device = g.device
  axis = g.uop.axis if isinstance(device, tuple) else None
  out_axis = None if axis is None else (1 if axis == 0 else 0)
  q = alloc_like((N, M), FP8_DTYPE, device, out_axis)
  e8 = alloc_like((N, M // BLK), dtypes.uint8, device, out_axis)
  fxn = functools.partial(_custom_transpose_quantize_mxfp8, dname=dname_of(device))
  q, e8, *_ = Tensor.custom_kernel(q, e8, g, fxn=fxn)
  return q, e8, mx_pack(e8)

def transpose_quantize_mxfp8_padded(g:Tensor, out_n:int) -> tuple[Tensor, Tensor, Tensor]:
  # Fused zero-pad + transpose + 1x32 MXFP8 quantization.  The source remains physically (M,src_n),
  # while q/e8 have the exact padded transposed shapes consumed by the GEMM.
  assert g.ndim == 2 and g.dtype == dtypes.bfloat16
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE, mx_pack
  M, src_n = g.shape
  assert M % BLK == 0 and src_n % BLK == 0 and src_n <= out_n and out_n % TILE_N == 0
  device, axis = g.device, (g.uop.axis if isinstance(g.device, tuple) else None)
  out_axis = None if axis is None else (1 if axis == 0 else 0)
  q = alloc_like((out_n, M), FP8_DTYPE, device, out_axis)
  e8 = alloc_like((out_n, M // BLK), dtypes.uint8, device, out_axis)
  q, e8, *_ = Tensor.custom_kernel(q, e8, g,
    fxn=functools.partial(_custom_transpose_quantize_mxfp8_padded, dname=dname_of(device)))
  return q, e8, mx_pack(e8)

def transpose_quantize_mxfp8_bias(g:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  # Columnwise MXFP8 plus FP32 sums for every 32-row tile, produced from the same BF16 loads.
  assert g.ndim == 2 and g.dtype == dtypes.bfloat16
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE, mx_pack
  M, N = g.shape
  out_n = ((N + TILE_N - 1) // TILE_N) * TILE_N
  assert M % BLK == 0 and N % BLK == 0
  device, axis = g.device, (g.uop.axis if isinstance(g.device, tuple) else None)
  out_axis = None if axis is None else (1 if axis == 0 else 0)
  q = alloc_like((out_n, M), FP8_DTYPE, device, out_axis)
  e8 = alloc_like((out_n, M // BLK), dtypes.uint8, device, out_axis)
  partial = alloc_like((M // BLK, out_n), dtypes.float32, device, axis)
  custom = _custom_transpose_quantize_mxfp8 if out_n == N else _custom_transpose_quantize_mxfp8_padded
  q, e8, partial, *_ = Tensor.custom_kernel(q, e8, partial, g,
    fxn=functools.partial(custom, dname=dname_of(device)))
  return q, e8, mx_pack(e8), partial

def transpose_quantize_mxfp8_bias_dual(g:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
  # In addition to the columnwise MXFP8 and bias partials, emit the rowwise MXFP8 layout from the same BF16 tile.
  # This is the pair of layouts consumed by dense MXFP8 dgrad/wgrad and avoids rereading g for generic row quantization.
  assert g.ndim == 2 and g.dtype == dtypes.bfloat16
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE, mx_pack
  M, N = g.shape
  out_n = ((N + TILE_N - 1) // TILE_N) * TILE_N
  assert M % BLK == 0 and N % BLK == 0 and out_n % (4 * BLK) == 0
  device, axis = g.device, (g.uop.axis if isinstance(g.device, tuple) else None)
  local_m = g.uop.shard_shape[0] if isinstance(device, tuple) and axis == 0 else M
  col_axis = None if axis is None else (1 if axis == 0 else 0)
  row_axis = axis
  si_axis = None if axis is None else (1 if axis == 0 else 0)
  col_q = alloc_like((out_n, M), FP8_DTYPE, device, col_axis)
  col_si_alias = (local_m, N, out_n) == (16384, 5120, 5120) and bool(getenv("GPTOSS_QKV_TQ_COL_E8_SI_ONLY", 0))
  col_e8 = alloc_like((M // (4 * BLK), out_n), dtypes.uint32, device, axis) if col_si_alias else \
    alloc_like((out_n, M // BLK), dtypes.uint8, device, col_axis)
  partial = alloc_like((M // BLK, out_n), dtypes.float32, device, axis)
  row_q = alloc_like((M, out_n), FP8_DTYPE, device, row_axis)
  row_si = alloc_like((out_n // (4 * BLK), M), dtypes.uint32, device, si_axis)
  row_e8_alias = (local_m, N, out_n) == (16384, 5120, 5120) and bool(getenv("GPTOSS_QKV_TQ_ROW_E8_SI_ONLY", 0))
  row_e8 = row_si if row_e8_alias else alloc_like((M, out_n // BLK), dtypes.uint8, device, row_axis)
  if row_e8_alias:
    col_q, col_e8, partial, row_q, row_si = (owned_empty(t) for t in (col_q, col_e8, partial, row_q, row_si))
    row_e8 = row_si  # Preserve the duplicate ABI argument's exact destination identity.
  custom = _custom_transpose_quantize_mxfp8 if out_n == N else _custom_transpose_quantize_mxfp8_padded
  col_q, col_e8, partial, row_q, row_e8, row_si, *_ = Tensor.custom_kernel(
    col_q, col_e8, partial, row_q, row_e8, row_si, g, fxn=functools.partial(custom, dname=dname_of(device)))
  return col_q, col_e8, col_e8 if col_si_alias else mx_pack(col_e8), partial, row_q, row_e8, row_si

def transpose_requantize_mxfp8(q:Tensor, e8:Tensor, packed_si:Tensor|None=None) -> tuple[Tensor, Tensor, Tensor]:
  # Fused dequant(q,e8).transpose(-2,-1)+requantize. The output rank matches the input rank and the final output
  # allocation has exactly the transposed shape consumed by GEMM (the 2D path does not rely on squeeze/view cleanup).
  assert q.ndim in (2, 3) and e8.ndim == q.ndim and q.dtype in dtypes.fp8s and e8.dtype == dtypes.uint8, \
    f"{q.shape} {q.dtype}, {e8.shape} {e8.dtype}"
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE
  is_2d = q.ndim == 2
  E, M, N = (1, *q.shape) if is_2d else q.shape
  assert e8.shape == ((M, N // BLK) if is_2d else (E, M, N // BLK)), f"{q.shape} {e8.shape}"
  assert packed_si is None or packed_si.shape == ((N // (4 * BLK), M) if is_2d else (E, N // (4 * BLK), M)), \
    f"bad packed scale input {packed_si.shape}"
  assert M % (4 * BLK) == 0 and N % TILE_N == 0, f"M={M} must%{4*BLK}, N={N} must%{TILE_N}"
  device = q.device
  axis = q.uop.axis if isinstance(device, tuple) else None
  q_axis = None if axis is None else ((1 - axis) if is_2d else (0 if axis == 0 else (2 if axis == 1 else 1)))
  si_axis = axis
  qt_shape = (N, M) if is_2d else (E, N, M)
  e8t_shape = (N, M // BLK) if is_2d else (E, N, M // BLK)
  si_shape = (M // (4 * BLK), N) if is_2d else (E, M // (4 * BLK), N)
  qt = alloc_like(qt_shape, FP8_DTYPE, device, q_axis)
  local_shape = q.uop.shard_shape if isinstance(device, tuple) else q.shape
  qkv_si_only = bool(getenv("GPTOSS_QKV_REQUANT_PACKED", 0) and getenv("GPTOSS_QKV_REQUANT_SI_ONLY", 0) and
                     packed_si is not None and not is_2d and local_shape == (1, 16384, 3072))
  e8t = e8 if qkv_si_only else alloc_like(e8t_shape, dtypes.uint8, device, q_axis)
  si = alloc_like(si_shape, dtypes.uint32, device, si_axis)
  fxn = functools.partial(_custom_transpose_requantize_mxfp8, dname=dname_of(device))
  inputs = ((packed_si,) if packed_si is not None else ()) + (q, e8)
  if qkv_si_only: qt, si = owned_empty(qt), owned_empty(si)
  qt, e8t, si, *_ = Tensor.custom_kernel(qt, e8t, si, *inputs, fxn=fxn)
  return qt, e8t, si

def transpose_requantize_mxfp8_dgrad(q:Tensor, e8:Tensor, packed_si:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  """GPT-OSS FC1 dgrad layout: emit transposed q+SI and retain source e8 only as the grouped-GEMM's unused ABI arg."""
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE
  assert q.shape == (32, 5888, 3072) and e8.shape == (32, 5888, 96)
  assert packed_si.shape == (32, 24, 5888) and packed_si.dtype == dtypes.uint32
  device, axis = q.device, q.uop.axis if isinstance(q.device, tuple) else None
  qt = alloc_like((32, 3072, 5888), FP8_DTYPE, device, None if axis is None else (0 if axis == 0 else 2 if axis == 1 else 1))
  si = alloc_like((32, 46, 3072), dtypes.uint32, device, axis)
  fxn = functools.partial(_custom_transpose_requantize_mxfp8, dname=dname_of(device))
  qt, e8_dep, si, *_ = Tensor.custom_kernel(owned_empty(qt), e8, owned_empty(si), packed_si, q, e8, fxn=fxn)
  return qt, e8_dep, si

def transpose_requantize_mxfp8_down_dgrad(q:Tensor, e8:Tensor, packed_si:Tensor|None=None) -> tuple[Tensor, Tensor, Tensor]:
  """GPT-OSS down dgrad layout: split M32 workgroups, reuse packed input scales, and omit its unused e8 output."""
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE
  assert q.shape == (32, 3072, 3072) and e8.shape == (32, 3072, 96)
  assert packed_si is None or (packed_si.shape == (32, 24, 3072) and packed_si.dtype == dtypes.uint32)
  device, axis = q.device, q.uop.axis if isinstance(q.device, tuple) else None
  qt = alloc_like((32, 3072, 3072), FP8_DTYPE, device, None if axis is None else (0 if axis == 0 else 2 if axis == 1 else 1))
  si = alloc_like((32, 24, 3072), dtypes.uint32, device, axis)
  fxn = functools.partial(_custom_transpose_requantize_mxfp8, dname=dname_of(device), down_dgrad=True)
  inputs = ((packed_si,) if packed_si is not None else ()) + (q, e8)
  qt, e8_dep, si, *_ = Tensor.custom_kernel(owned_empty(qt), e8, owned_empty(si), *inputs, fxn=fxn)
  return qt, e8_dep, si
