import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.helpers import getenv
from tinygrad.renderer import Estimates
from extra.hipcc import HIPCCCompiler, rocm_toolchain, hip_program
from extra.llama_kernels import alloc_like, kernel_grad
from extra.gemm.cdna_asm_gemm import quantize_mxfp8, _mx_block_scale, _mx_block_scale_3d, mx_pack, FP8_DTYPE

# ZeRO-2 keeps the DP gradient axis-zero sharded; disabling it retains all-reduce.
ZERO2 = getenv("ZERO2", 0)
# FC1's 128x256 tile reuses the packed 256-row scale super-tile; SCALE_PACK is independent.
TILE_M128 = getenv("TILE_M128", 0)
# Read raw row-major e8 scales and pack them inside GEMM instead of calling mx_pack/mx_pack_3d.
SCALE_PACK = getenv("SCALE_PACK", 0)
# Emit post-SwiGLU columnwise MXFP8 for FC2 wgrad. It quantizes dequant(rowwise y_fp8),
# matching transpose_quantize_mxfp8; NATIVE_MXFP8_CVT must agree with that consumer.
FUSED_FC1_COLW = getenv("FUSED_FC1_COLW", 0)
NATIVE_MXFP8_CVT = getenv("NATIVE_MXFP8_CVT", 0)
# Optional optimizer-owned transposed weights replace backward dequantize/transpose/requantize.
# Forward wrappers register (wT_q, wT_e8) under w_q.uop for _dgrad_wT_operand.
PRESTORE_WT = getenv("PRESTORE_WT", 0)
# Fold FC2 dgrad + dSwiGLU + rowwise mxfp8 quantization into one grouped GEMM. The epilogue consumes the saved
# FC1 pre-activation and emits d_h directly, eliminating the bf16 d_y round-trip and standalone dSwiGLU kernel.
FUSED_DGRAD_DSWIGLU = getenv("FUSED_DGRAD_DSWIGLU", 0)
# Columnwise dH feeds FC1 wgrad; NATIVE_MXFP8_CVT must match standalone transpose-quantize.
FUSED_DGRAD_COLW = getenv("FUSED_DGRAD_COLW", 0)
# GPT-OSS split-dgrad companion: dSwiGLU directly produces rowwise+columnwise dH MXFP8 and 32-row dBias partials.
# This removes both the dead full-BF16 dH store and transpose_quantize(dH). Exact shape only, AMD, default on.
FUSED_DH_DUAL_PRODUCER = getenv("FUSED_DH_DUAL_PRODUCER", 1)
FUSED_DH_BIAS_PARTIAL = getenv("FUSED_DH_BIAS_PARTIAL", 0)

# Backward-to-backward gradient companions. Forward FC1 checkpoints are recovered from the explicit CALL ABI below.
_fused_dswiglu_grad_mailbox: dict = {}

# Forward-to-backward handoff of optimizer-owned transposed weights.
_dgrad_wT_mailbox: dict = {}
def _dgrad_wT_operand(w_uop:UOp) -> tuple[Tensor, Tensor]|None:
  if not PRESTORE_WT or not _dgrad_wT_mailbox: return None
  if (hit := _dgrad_wT_mailbox.get(w_uop)) is not None: return hit
  b = w_uop.base if w_uop is not w_uop.base else None
  return _dgrad_wT_mailbox.get(b) if b is not None else None

def _dynamic_dgrad_wT_operand(w_q:Tensor, w_e8:Tensor, w_si:Tensor|None=None) -> Tensor|tuple[Tensor, Tensor, Tensor]:
  _, M, N = w_q.shape
  if M % 128 == 0 and N % 256 == 0:
    from extra.llama_kernels.transpose_quantize_mxfp8 import (transpose_requantize_mxfp8,
      transpose_requantize_mxfp8_dgrad, transpose_requantize_mxfp8_down_dgrad)
    # The GPT-OSS optimizer maintains w_gate_up_si as persistent companion state and updates it alongside w_e8.
    # Reuse that packed layout here; this avoids replaying a pack and gives requantize contiguous scale reads over M.
    use_si = w_si is not None and bool(getenv("PREPACK_FC1_WSI", 0)) and bool(getenv("GPTOSS_REQUANT_PACKED_INPUT", 1)) \
      and tuple(w_q.shape) == (32, 5888, 3072)
    if use_si:
      assert w_si.dtype == dtypes.uint32 and w_si.shape == (32, 24, 5888)
      if bool(getenv("GPTOSS_REQUANT_SI_ONLY", 1)): return transpose_requantize_mxfp8_dgrad(w_q, w_e8, w_si)
    if (M, N) == (3072, 3072) and bool(getenv("GPTOSS_DOWN_REQUANT", 1)):
      down_si = w_si if w_si is not None and w_si.dtype == dtypes.uint32 and w_si.shape == (32, 24, 3072) and \
        bool(getenv("GPTOSS_DOWN_REQUANT_PACKED_INPUT", 1)) else None
      return transpose_requantize_mxfp8_down_dgrad(w_q, w_e8, down_si)
    return transpose_requantize_mxfp8(w_q, w_e8, w_si if use_si else None)
  return (w_q.cast(dtypes.bfloat16) * _mx_block_scale_3d(w_e8).cast(dtypes.bfloat16)).transpose(1, 2)

def _fused_dswiglu_lookup(gradient:UOp) -> tuple[UOp, ...]|None:
  if not FUSED_DGRAD_DSWIGLU or not _fused_dswiglu_grad_mailbox: return None
  seen, stack = set(), [gradient]
  while stack and len(seen) < 128:
    u = stack.pop()
    if id(u) in seen: continue
    seen.add(id(u))
    if (hit := _fused_dswiglu_grad_mailbox.get(u)) is not None: return hit
    if u is not u.base and (hit := _fused_dswiglu_grad_mailbox.get(u.base)) is not None: return hit
    if u.op in (Ops.AFTER, Ops.PAD, Ops.RESHAPE, Ops.SHRINK, Ops.STAGE, Ops.EXPAND, Ops.CAST, Ops.PERMUTE):
      stack.extend(u.src)
  return None

def _finish_expert_wgrad(t:Tensor) -> Tensor:
  # The reduced GPT-OSS shard already has the backward boundary's exact layout. Keep apply_grad's owned
  # destination unchanged, but avoid an extra identity copy before returning the reduced gradient.
  if t.shape in ((32, 5888, 3072), (32, 3072, 3072)) and t.dtype == dtypes.bfloat16 and \
      isinstance(t.device, tuple) and len(t.device) == 8 and t.uop.axis == 0: return t
  return t.contiguous()

def reduce_scatter_devaxis(out:Tensor, shard_axis:int=0) -> Tensor:
  # out: multi tensor device-sharded on axis 0 (1 slice/device), logical shape (ndev, *rest); each device holds
  # its partial. Returns sum over the device axis, left SHARDED on `shard_axis` of the *rest result. Clean all2all:
  # device i pulls ONLY its shard-rows from every device's physical partial (via mselect, no device-0 hub) and
  # sums -> 0.875x per-device inbound, all xGMI links (vs 1.75x for a full all-reduce).
  u = out.uop
  devs = u.device
  ndev = len(devs)
  rest = u.shape[1:]
  ax = shard_axis
  full = rest[ax]
  assert full % ndev == 0, f"reduce_scatter needs even shards: {full} % {ndev}"
  sz = full // ndev
  # peel to the UNSHARD carrying the device sharding; its src[0] is the raw per-device multi-buffer whose
  # mselect(j) is device j's PHYSICAL partial (shape (1, *rest)) -- extracting via shrink on the device axis
  # would instead gather every partial onto device 0. Re-attach the kernel-write barrier(s) that sit on the
  # AFTER node(s) above the UNSHARD, else mselect reads the pre-write (invalids) buffer -> nan.
  node, barriers = u, []
  while node.op is not Ops.UNSHARD:
    if node.op is Ops.AFTER: barriers += node.src[1:]
    node = node.src[0]
  mbuf = node.src[0]
  if barriers: mbuf = mbuf.after(*barriers)
  rng = UOp.range(ndev, -1, AxisType.DEVICE)
  shards = []
  for i in range(ndev):
    shr = tuple((0,s) if a != ax else (i*sz,(i+1)*sz) for a,s in enumerate(rest))
    contribs = [mbuf.mselect(j).reshape(rest).shrink(shr).copy_to_device(devs[i]) for j in range(ndev)]
    shards.append(functools.reduce(lambda a,b: a.alu(Ops.ADD, b), contribs))
  return Tensor(UOp.mstack(*shards).unshard(ax, rng), device=devs)

def _reduce_scatter_expert_chunks(chunks:list[Tensor]) -> Tensor:
  if getenv("GPTOSS_DEFER_FC1_REDUCE", 0) and chunks[0].shape[-2:] == (4*5888, 3072):
    # An explicit application boundary: the layer backward can return the producer's
    # eight physical chunks and leave this collective in its caller's stock schedule.
    return _deferred_expert_chunks(chunks)
  return _sum_expert_chunks(chunks)

def _deferred_expert_chunks(chunks:list[Tensor]) -> Tensor:
  args = tuple(t.uop for t in chunks)
  value = _sum_expert_chunks([Tensor(t.param_like(i)) for i,t in enumerate(args)]).uop
  assert value.op is Ops.UNSHARD and value.src[0].op is Ops.MSTACK and len(value.src[0].src) == 8
  result = value.call_with_output(*args, name="gptoss_fc1_reduce", aux="gptoss_fc1_reduce")
  call, destination = result.src[1], value.param_like(len(args)).src[0]
  # MSTACK materializes its arithmetic inputs. Inline per-device stores instead
  # write the owned output directly, retaining the exact BF16 addition tree.
  stores = []
  for i,part in enumerate(value.src[0].src):
    dst, inputs = destination.mselect(i), []
    def flatten_sum(u):
      if u.op is Ops.ADD:
        flatten_sum(u.src[0])
        flatten_sum(u.src[1])
      else: inputs.append(u)
    flatten_sum(part)
    assert len(inputs) == 8
    formal = tuple(x.param_like(j+1) for j,x in enumerate(inputs))
    expression = part.substitute(dict(zip(inputs,formal)), walk=True)
    # STORE destinations stay PARAMs; normal inline-call binding selects each lane.
    stores.append(dst.param_like(0).store(expression).sink().call(dst, *inputs))
  return Tensor(result.replace(src=(result.src[0], call.replace(src=(UOp.sink(*stores), *call.src[1:])))))

def _sum_expert_chunks(chunks:list[Tensor]) -> Tensor:
  # Wgrad writes each destination's four experts into a separate allocation. Peer copies can use that whole
  # allocation directly, avoiding the contiguous staging kernels required by slices of the full expert buffer.
  devs = chunks[0].device
  assert isinstance(devs, tuple) and len(chunks) == len(devs) == 8
  raw = []
  for t in chunks:
    assert t.device == devs and t.uop.axis == 0 and t.shape == chunks[0].shape
    node, barriers = t.uop, []
    while node.op is not Ops.UNSHARD:
      if node.op is Ops.AFTER: barriers.extend(node.src[1:])
      node = node.src[0]
    raw.append(node.src[0].after(*barriers) if barriers else node.src[0])
  reduced = []
  for i, device in enumerate(devs):
    inputs = [raw[i].mselect(j).copy_to_device(device) for j in range(len(devs))]
    # Preserve reduce_scatter_devaxis's rank order and BF16 rounding after each addition.
    reduced.append(functools.reduce(lambda a,b: a.alu(Ops.ADD, b), inputs))
  return Tensor(UOp.mstack(*reduced).unshard(0, UOp.range(len(devs), -1, AxisType.DEVICE)), device=devs)

def custom_hk_grouped_mxfp8_gemm(C:UOp, A:UOp, B:UOp, scale_A:UOp, scale_B:UOp, *extra:UOp, dname:str, n_experts:int,
                                  logical_k:int|None=None) -> UOp:
  M, K = A.shape
  E, N, K2 = B.shape
  assert K == K2, f"{A.shape} {B.shape}"
  assert E == n_experts, f"{E} != {n_experts}"
  assert len(extra) in (3, 4), f"expected (a_e8,b_e8,expert_off[,out_scale_e8]), got {len(extra)} extra inputs"
  fused_out_scale = len(extra) == 4
  packed_out_scale = fused_out_scale and extra[3].dtype == dtypes.uint32
  if fused_out_scale:
    if packed_out_scale:
      assert (E, M, N, K) == (32, 73728, 3072, 5888) and extra[3].shape == (N // 128, M), \
        f"packed GPT-OSS out_scale must be uint32 ({N // 128},{M}), got {extra[3].dtype} {extra[3].shape}"
    else:
      assert extra[3].shape == (M, N // 32) and extra[3].dtype == dtypes.uint8, \
        f"out_scale_e8 must be uint8 ({M},{N // 32}), got {extra[3].dtype} {extra[3].shape}"
  gptoss_dgrad_tail = bool(getenv("GPTOSS_DGRAD_TAIL", 0)) and fused_out_scale and (E, M, N, K) == (32, 73728, 3072, 5888)
  # The GPT-OSS down gradient is physically padded from 2880 to 3072 columns. Its final 192 columns are zero, so
  # the down DGRAD can stop after the 23rd 128-wide tile while retaining the physical strides and scale pitches.
  gptoss_down_logical_k = logical_k == 2880 and not fused_out_scale and (E, M, N, K) == (32, 73728, 3072, 3072) \
    and bool(getenv("GPTOSS_DOWN_DGRAD_LOGICAL_K", 1))
  # The same exact down DGRAD has only 2880 live output columns. On its final 256-column tile, load/compute only
  # the first 64 columns and explicitly zero the remaining physical ABI. This implementation relies on the DBUF
  # loop's guarded second-B load and is kept independently disableable.
  gptoss_down_out_tail = gptoss_down_logical_k and bool(getenv("DBUF", 1)) and bool(getenv("GPTOSS_DOWN_DGRAD_OUT_TAIL", 1))
  # The 23rd contraction tile contains only 64 live values. This keeps the required x128 MFMA (with the upper
  # operand registers zero) but skips the unused global/LDS staging and upper-half LDS reads. Fused down+bias is
  # excluded because the same control expansion spills there; that callback passes this source switch as zero.
  gptoss_down_k64_tail = gptoss_down_out_tail and bool(getenv("GPTOSS_DOWN_DGRAD_K64_TAIL", 1))
  # The exact down dgrad's ping-pong plane index otherwise materializes the eight LDS bases in 16 KiB of LDS.
  # Select from the scalar bases directly; exact poisoned/repeat validation passed and this is independently gated.
  gptoss_down_scalar_lds = gptoss_down_k64_tail and bool(getenv("GPTOSS_DOWN_DGRAD_SCALAR_LDS", 1))
  # Inactive expert-row tiles must still initialize the physical output. On this exact path, use aligned 16-byte
  # zero stores instead of scalar bf16 stores. Keep it coupled to the fully specialized down-dInput path so the
  # shared grouped GEMM's generic/fused-bias ABIs remain byte-for-byte unchanged.
  gptoss_down_skip_empty_vec = gptoss_down_k64_tail and bool(getenv("MOE_SKIP_EMPTY", 0)) and \
    not bool(getenv("MOE_SKIP_NO_ZERO", 0)) and bool(getenv("GPTOSS_DOWN_DGRAD_SKIP_EMPTY_VEC", 1))
  # The exact non-bias down dInput maps its 288 row tiles more evenly with twelve-row groups. Keep the library
  # default unchanged; the GPT-OSS launch script selects the qualified value and the kernel name records its ABI.
  down_dgrad_wgm = getenv("GPTOSS_DOWN_DGRAD_WGM", 8) if gptoss_down_k64_tail else 8
  # This exact down dInput uses a 32-row B fragment, so only packed scale bytes 0/1 are selected by its MFMAs.
  # Reuse the existing two-byte helper; keep other grouped/FC1 paths and disabled launch names unchanged.
  down_dgrad_pack_b32 = bool(getenv("GPTOSS_DOWN_DGRAD_PACK_B32", 0)) and gptoss_down_scalar_lds and \
    gptoss_down_skip_empty_vec and down_dgrad_wgm == 12
  # Runtime expert counts can skip row tiles; M remains a capacity upper bound, not an exact routing measurement.
  executed_k = 5760 if gptoss_dgrad_tail else 2944 if gptoss_down_logical_k else K
  executed_n = 2880 if (gptoss_dgrad_tail and getenv("DBUF", 1)) or gptoss_down_out_tail else N
  dgrad_rocm72_requested = gptoss_dgrad_tail and bool(getenv("GPTOSS_DGRAD_ROCM72", 0))
  dgrad_rocm72_hipcc, dgrad_rocm72_root = rocm_toolchain("/opt/rocm-7.2.1", dgrad_rocm72_requested,
    flag="GPTOSS_DGRAD_ROCM72")
  dgrad_rocm72 = dgrad_rocm72_root is not None
  threads = UOp.special(64 * 8, "lidx0")
  workgroups = UOp.special((M // 256) * (N // 256), "gidx0")
  sink_inputs = (C.base, A.base, B.base, scale_A.base, scale_B.base, *(x.base for x in extra), threads, workgroups)
  # This precompiled kernel only exposes its pointer arguments to the UOp graph. Describe the real accesses so
  # captured HCQ graphs order consumers/copies after C instead of treating every pointer as a read-only input.
  zero = UOp.const(0, dtypes.int32)
  accesses = (C.index(zero).store(UOp.const(0, C.dtype)),) + \
    tuple(x.index(zero).load() for x in (A, B, scale_A, scale_B, *extra))
  # Five upper-bound probes replace the linear 31-offset scan for exact GPT-OSS DGRAD.
  dgrad_binary_expert = gptoss_dgrad_tail and bool(getenv("GPTOSS_DGRAD_BINARY_EXPERT", 1))
  # Exact FC1 dgrad: select each double-buffered LDS plane from literal scalar bases instead of runtime-indexed
  # address arrays. Library default remains off; the GPT-OSS submission explicitly opts into the qualified path.
  dgrad_scalar_lds = gptoss_dgrad_tail and bool(getenv("DBUF", 1)) and bool(getenv("GPTOSS_DGRAD_SCALAR_LDS", 0))
  sink = UOp.sink(*sink_inputs, *accesses,
                  arg=KernelInfo(
                    f"hk_{'gptoss_dgrad_tail' if gptoss_dgrad_tail else 'grouped_mxfp8_gemm_outscale' if fused_out_scale else 'grouped_mxfp8_gemm'}"+
                    (f"_bin{int(dgrad_binary_expert)}" if gptoss_dgrad_tail else "")+
                    (f"_slds{int(dgrad_scalar_lds)}" if gptoss_dgrad_tail else "")+
                    (f"_r72{int(dgrad_rocm72)}" if dgrad_rocm72_requested else "")+
                    (f"_down_logicalk{int(gptoss_down_logical_k)}" if logical_k is not None else "")+
                    (f"_outtail{int(gptoss_down_out_tail)}" if gptoss_down_logical_k else "")+
                    (f"_k64tail{int(gptoss_down_k64_tail)}" if gptoss_down_out_tail else "")+
                    (f"_slds{int(gptoss_down_scalar_lds)}" if gptoss_down_k64_tail else "")+
                    (f"_evec{int(gptoss_down_skip_empty_vec)}" if gptoss_down_k64_tail else "")+
                    (f"_wgm{down_dgrad_wgm}" if gptoss_down_k64_tail else "")+
                    ("_b32" if down_dgrad_pack_b32 else "")+
                    f"_{E}_{M}_{N}_{K}", estimates=Estimates(ops=2*M*executed_n*executed_k,
                    mem=(M*K+E*N*K)*A.dtype.itemsize+M*N*C.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens_path/"grouped_mxfp8_gemm.cpp").read_text()
  # Prefetch the next K tile while computing the current one, retaining the accumulation order.
  dbuf = ["-DDOUBLE_BUFFER=1"] if getenv("DBUF", 1) else []
  grouped_wgm = getenv("GPTOSS_DGRAD_WGM", 2) if gptoss_dgrad_tail else \
    down_dgrad_wgm if gptoss_down_k64_tail else getenv("GROUPED_WGM", 8)
  dgrad_pack_b32 = (bool(getenv("GPTOSS_DGRAD_PACK_B32", 1)) and gptoss_dgrad_tail) or down_dgrad_pack_b32
  dgrad_stage_out_scale = getenv("DGRAD_STAGE_OUT_SCALE", 0)
  dgrad_scale_broadcast_lds = gptoss_dgrad_tail and packed_out_scale and dgrad_stage_out_scale == 2 and \
    bool(getenv("GPTOSS_DGRAD_SCALE_BROADCAST_LDS", 1))
  dgrad_split_lgkm = gptoss_dgrad_tail and bool(getenv("DBUF", 1)) and bool(getenv("GPTOSS_DGRAD_SPLIT_LGKM", 1))
  lib = HIPCCCompiler("gfx950", [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}",
                                 f"-DGEMM_E={E}", f"-DFUSED_OUT_SCALE={int(fused_out_scale)}",
                                 f"-DFUSED_OUT_SCALE_PACKED={int(packed_out_scale)}",
                                 f"-DDGRAD_STAGE_OUT_SCALE={dgrad_stage_out_scale}",
                                 f"-DGPTOSS_DGRAD_TAIL={int(gptoss_dgrad_tail)}",
                                 f"-DGPTOSS_DOWN_LOGICAL_K={int(gptoss_down_logical_k)}", "-DGPTOSS_DOWN_REAL_K=2880",
                                 f"-DGPTOSS_DOWN_DGRAD_OUT_TAIL={int(gptoss_down_out_tail)}", "-DGPTOSS_DOWN_REAL_N=2880",
                                 f"-DGPTOSS_DOWN_DGRAD_K64_TAIL={int(gptoss_down_k64_tail)}",
                                 f"-DGPTOSS_DOWN_DGRAD_SCALAR_LDS={int(gptoss_down_scalar_lds)}",
                                 f"-DGPTOSS_DOWN_DGRAD_SKIP_EMPTY_VEC={int(gptoss_down_skip_empty_vec)}",
                                 f"-DGPTOSS_DGRAD_PACK_B32={int(dgrad_pack_b32)}",
                                 f"-DGPTOSS_DGRAD_BINARY_EXPERT={int(dgrad_binary_expert)}",
                                 f"-DGPTOSS_DGRAD_SCALAR_LDS={int(dgrad_scalar_lds)}",
                                 # Dedicated mode-2 scale LDS supports same-address subgroup broadcasts; the fixed
                                 # output-scale read no longer needs a leader-lane EXEC mask plus ds_bpermute.
                                 f"-DGPTOSS_DGRAD_SCALE_BROADCAST_LDS={int(dgrad_scale_broadcast_lds)}",
                                 f"-DGPTOSS_DGRAD_SPLIT_LGKM={int(dgrad_split_lgkm)}",
                                 f"-DMOE_SKIP_EMPTY={getenv('MOE_SKIP_EMPTY', 0)}", f"-DMOE_SKIP_NO_ZERO={getenv('MOE_SKIP_NO_ZERO', 0)}",
                                 f"-DGROUPED_WGM={grouped_wgm}", *dbuf],
                      hipcc_path=dgrad_rocm72_hipcc,
                      rocm_path=dgrad_rocm72_root).compile_cached(src)
  return hip_program(sink, src, lib)

@functools.cache
def custom_hk_grouped_mxfp8_gemm_swiglu(H:UOp, Y:UOp, Ye8:UOp, Bias:UOp, A:UOp, B:UOp, scale_A:UOp, scale_B:UOp, *extra:UOp,
                                        dname:str, n_experts:int, real_inter:int, has_expert_counts:bool=False) -> UOp:
  # fused FC1: gate_up = A @ B^T (M, N), add per-expert Bias (E, N) bf16, emit the pre-swiglu H (M, N) bf16 [saved
  # for the backward] plus SwiGLU + mxfp8-quantize in the epilogue -> Y (M, N//2) fp8e4m3 block-scaled with Ye8
  # (M, N//64) e8m0. Never round-trips gate_up through HBM.
  M, K = A.shape
  E, N, K2 = B.shape
  assert K == K2, f"{A.shape} {B.shape}"
  assert E == n_experts, f"{E} != {n_experts}"
  assert len(Bias.shape) == 2 and Bias.shape[0] == E, f"bias must have E={E} rows, got {Bias.shape}"
  bias_n = Bias.shape[1]
  inter = N // 2
  out_inter = Y.shape[1]
  gptoss_shape = (E, M, N, K) == (32, 73728, 5888, 3072)
  fwd_rocm72_requested = gptoss_shape and bool(getenv("GPTOSS_FC1_FWD_ROCM72", 0))
  fwd_rocm72_hipcc, fwd_rocm72_root = rocm_toolchain("/opt/rocm-7.2.1", fwd_rocm72_requested,
    flag="GPTOSS_FC1_FWD_ROCM72")
  fwd_rocm72 = fwd_rocm72_root is not None
  skip_empty_main_req = bool(getenv("FC1_SKIP_EMPTY_MAIN_ZERO", 0))
  gptoss_skip_empty_main = skip_empty_main_req and gptoss_shape and bool(getenv("MOE_SKIP_EMPTY", 0)) \
    and FUSED_FC1_COLW and FUSED_DGRAD_DSWIGLU and FUSED_DH_DUAL_PRODUCER \
    and bool(getenv("FUSED_DGRAD_DSWIGLU_SPLIT", 0)) and not bool(getenv("FUSED_DOWN_EPILOGUE", 0))
  warp_scale32_req = getenv("FC1_WARP_SCALE32", 1)
  gptoss_warp_scale32 = bool(warp_scale32_req) and (gptoss_shape or warp_scale32_req == 2)
  full_lds_req = getenv("FC1_FULL_LDS_TILES", 1)
  gptoss_full_lds = bool(full_lds_req) and (gptoss_shape or full_lds_req == 2)
  fixed_wgm_req = getenv("FC1_FIXED_WGM_MAP", 1)
  gptoss_fixed_wgm = bool(fixed_wgm_req) and (gptoss_shape or fixed_wgm_req == 2)
  vector_tail_zero_req = getenv("FC1_VECTOR_TAIL_ZERO", 1)
  gptoss_vector_tail_zero = bool(vector_tail_zero_req) and (gptoss_shape or vector_tail_zero_req == 2)
  # The production split dSwiGLU walks only REAL_INTER H pairs and initializes padded dH itself. In that exact
  # configuration the saved padded H tail is dead, so FC1 can leave it unwritten. Keep every alternate backward
  # implementation on the initialized-H ABI until it independently provides the same guarantee.
  skip_h_tail_zero_req = getenv("FC1_SKIP_H_TAIL_ZERO", 1)
  gptoss_skip_h_tail_zero = bool(skip_h_tail_zero_req) and gptoss_shape and FUSED_DGRAD_DSWIGLU \
    and bool(getenv("FUSED_DGRAD_DSWIGLU_SPLIT", 0)) and bool(getenv("DSWIGLU_REAL_ONLY_TAIL", 1)) \
    and not getenv("DSWIGLU_FAST_MATH", 0)
  vector_empty_zero_req = getenv("FC1_VECTOR_EMPTY_ZERO", 0)
  gptoss_vector_empty_zero = bool(vector_empty_zero_req) and (gptoss_shape or vector_empty_zero_req == 2)
  zero_nt_req = getenv("FC1_ZERO_NT", 1)
  gptoss_zero_nt = bool(zero_nt_req) and (gptoss_shape or zero_nt_req == 2)
  med3_req = getenv("FC1_MED3_CLAMP", 1)
  gptoss_med3 = bool(med3_req) and (gptoss_shape or med3_req == 2)
  # GPT-OSS has exactly 32 experts and monotonic padded offsets. Binary lifting finds the owning expert with five
  # uniform offset probes instead of the generic 31-entry linear scan.
  binary_expert_req = getenv("FC1_BINARY_EXPERT", 1)
  gptoss_binary_expert = bool(binary_expert_req) and (gptoss_shape or binary_expert_req == 2)
  # gfx950 can exchange the fixed lane^1 SwiGLU partner with ds_swizzle instead of a generic shuffle. Full five-output
  # poison/repeat checks are exact with unchanged 256 VGPR/75 SGPR/132KB LDS; ABBA64 is 0.3-0.6% faster on GPUs2/3.
  # Complete production JITBEAM=3 A/B also wins on both GPUs; retain an opt-out for diagnostics.
  gptoss_ds_swizzle_xor1 = gptoss_shape and bool(getenv("GPTOSS_FC1_DS_SWIZZLE", 1))
  # Exact GPT-OSS FC1 pads hidden 2880 -> 3072 with zeros; 23 MFMA blocks cover the logical width, so the 24th
  # block is provably dead. Full five-output poison/repeat checks are byte-exact; ABBA64 saves 2.1-2.6% on GPUs2/3.
  # Keep generic shapes on their full physical K and retain an override for diagnostics.
  skip_pad_k_req = getenv("FC1_SKIP_PAD_K", 1)
  gptoss_real_k = 2880 if skip_pad_k_req and gptoss_shape else K
  executed_k = ((gptoss_real_k + 127) // 128) * 128
  # The exact GPT-OSS contraction has 23 x128 tiles. Peeling its final tile removes both has-next predicates from
  # the 22-iteration steady state without changing MFMA order; keep non-GPTOSS shapes on the generic loop.
  gptoss_peel_final = gptoss_shape and bool(getenv("FC1_PEEL_FINAL", 0))
  # Dispatch already produces the true unpadded expert counts. For each expert whose last 256-row block has only
  # a live upper half, keep the normal bias/SwiGLU epilogue but omit the lower-half A LDS read and cC/cD MFMAs.
  # Full five-output poison/repeat checks are byte-exact; callback-only ABBA64 saves 3.8-4.1% on GPUs2/3.
  skip_pad_half_req = getenv("FC1_SKIP_PAD_HALF_ROWS", 1)
  gptoss_skip_pad_half = bool(skip_pad_half_req) and gptoss_shape and has_expert_counts
  pad_lower_reuse_req = getenv("FC1_PAD_LOWER_REUSE", 1)
  gptoss_pad_lower_reuse = bool(pad_lower_reuse_req) and gptoss_skip_pad_half and gptoss_warp_scale32 \
    and FUSED_FC1_COLW and bool(getenv("FC1_DUAL_COLW_LDS", 1))
  # Coordinated GPT-OSS ABI: once dSwiGLU and both wgrad consumers use exact counts, a wholly padded lower 128-row
  # half is never read and the FC1 producer may omit its entire H/Y/scale/columnwise epilogue. Keep opt-in until the
  # complete backward chain has passed poisoned-buffer and convergence qualification.
  gptoss_omit_pad_lower_outputs = gptoss_pad_lower_reuse and bool(getenv("GPTOSS_FC1_OMIT_PAD_LOWER_OUTPUTS", 0))
  # With both dead MMA buffers holding the two 128-row transpose stripes, pair adjacent 32-token blocks per
  # thread and combine their e8m0 scale writes. This preserves the columnwise layout while halving scale stores.
  gptoss_colw_adj_scale2 = gptoss_shape and FUSED_FC1_COLW and gptoss_warp_scale32 \
    and bool(getenv("FC1_DUAL_COLW_LDS", 1)) and bool(getenv("GPTOSS_FC1_COLW_ADJ_SCALE2", 0))
  # Stage each row's four e8m0 bytes in the unused half of its columnwise LDS stripe, then coalesce the existing
  # scattered byte stores after the already-required epilogue barrier. The half-N tail writes only its live pair.
  gptoss_row_scale_lds4 = gptoss_shape and FUSED_FC1_COLW and gptoss_warp_scale32 \
    and bool(getenv("FC1_DUAL_COLW_LDS", 1)) and bool(getenv("GPTOSS_FC1_ROW_SCALE_LDS4", 0))
  # FC1 has its own placement knob: changing the process-wide GROUPED_WGM also retunes unrelated grouped GEMMs.
  # Keep FC1 placement independent from the process-wide GROUPED_WGM. With the coalesced row/column scale stores,
  # three-row groups plus x8/c64 win across balanced, skewed, and boundary-heavy GPT-OSS routing on two MI350X GPUs.
  # Generic shapes retain the shared WGM default, and WGM6 remains available as a diagnostic override.
  grouped_wgm = getenv("GPTOSS_FC1_WGM", 3) if gptoss_shape else getenv("GROUPED_WGM", 8)
  half_n_tail = bool(getenv("FC1_HALF_N_TAIL", 0)) and (2 * real_inter) % 256 == 128
  # GPT-OSS has 22 complete N tiles and one half tile. Instantiate the contraction loop once for each case behind
  # a uniform workgroup branch so the common interior path contains no half-tail predicates. Library default stays
  # off; the GPT-OSS submission opts into the independently qualified specialization.
  gptoss_interior_fast = gptoss_shape and half_n_tail and gptoss_full_lds and gptoss_peel_final \
    and bool(getenv("DBUF", 1)) and bool(getenv("GPTOSS_FC1_INTERIOR_FAST", 0))
  # On an expert's partial final row tile, a 32-row upper wave band beginning at or beyond the exact count has
  # untouched +0 accumulators. Keep the complete-upper-half contraction byte-identical and specialize only the
  # rare partial case, where the inactive band omits its A LDS read and cA/cB MFMAs.
  gptoss_pad_upper_skip = gptoss_interior_fast and gptoss_pad_lower_reuse \
    and bool(getenv("GPTOSS_FC1_PAD_UPPER_SKIP", 0))
  # Pass the exact interior path's precomputed byte offsets straight to its raw global-to-LDS loads. Keep the
  # library default off; the GPT-OSS submission enables the independently exact/timed specialization.
  gptoss_precomputed_soff = gptoss_interior_fast and bool(getenv("GPTOSS_FC1_PRECOMPUTED_SOFF", 0))
  # Load each exact 256-column bias tile once into dead scale LDS before the register epilogue. The existing
  # post-contraction barrier publishes it, and all four row warps reuse the staged values.
  gptoss_bias_lds = gptoss_interior_fast and gptoss_warp_scale32 and bool(getenv("GPTOSS_FC1_BIAS_LDS", 0))
  # On the exact WGM=3/6 `_ifast1` grid, assigning 64 consecutive workgroups across all eight XCD partitions gives
  # the best chiplet/L2 balance across balanced, skewed, and boundary-stress routing. Retain earlier mappings as
  # explicit diagnostic fallbacks and keep generic shapes on their ordinary permutation.
  gptoss_map_x8c64 = gptoss_interior_fast and gptoss_fixed_wgm and grouped_wgm in (3, 6) \
    and bool(getenv("GPTOSS_FC1_MAP_X8_C64", 1))
  gptoss_map_x4c48 = gptoss_interior_fast and gptoss_fixed_wgm and grouped_wgm == 6 \
    and not gptoss_map_x8c64 and bool(getenv("GPTOSS_FC1_MAP_X4_C48", 0))
  gptoss_map_x2c96 = gptoss_interior_fast and gptoss_fixed_wgm and grouped_wgm == 8 \
    and bool(getenv("GPTOSS_FC1_MAP_X2_C96", 0))
  assert bias_n == N or (half_n_tail and bias_n == 2 * real_inter), \
    f"compact bias width {bias_n} requires the matching half-N tail ({2 * real_inter})"
  assert Y.shape == (M, out_inter) and out_inter >= inter and out_inter % 32 == 0
  assert Ye8.shape == (M, out_inter // 32), f"bad padded row-scale output {Ye8.shape}"
  blk_row = 128 if TILE_M128 else 256
  threads = UOp.special(64 * 8, "lidx0")
  workgroups = UOp.special((M // blk_row) * (N // 256), "gidx0")
  count_args = (extra[-1].base,) if has_expert_counts else ()
  # M is a capacity upper bound: the exact number of row/half-row tiles depends on runtime expert counts.
  abi_extra = extra[:-1] if has_expert_counts else extra
  base_extra = 5 if FUSED_FC1_COLW else 3
  assert len(abi_extra) in (base_extra, base_extra + 2)
  # Keep active columnwise outputs at their original ABI offsets; dependency-only x-columnwise anchors are tail args.
  colw = (abi_extra[3].base, abi_extra[4].base) if FUSED_FC1_COLW else ()
  colw_packed = bool(colw) and abi_extra[4].dtype == dtypes.uint32
  if colw_packed:
    assert gptoss_shape and out_inter == 3072 and abi_extra[4].shape == (M // 128, out_inter)
  xanchors = (abi_extra[-2].base, abi_extra[-1].base) if len(abi_extra) == base_extra + 2 else ()
  sink_inputs = (H.base, Y.base, Ye8.base, Bias.base, A.base, B.base, scale_A.base, scale_B.base,
                 abi_extra[0].base, abi_extra[1].base, abi_extra[2].base, *colw, *xanchors, *count_args, threads, workgroups)
  # Metadata-only accesses preserve the compiled HIP ABI while making all five optional producer outputs visible
  # to HCQ dependency analysis. The trailing x-columnwise buffers are dependency-only inputs, never outputs.
  zero = UOp.const(0, dtypes.int32)
  output_bufs = (H, Y, Ye8, *extra[3:5]) if FUSED_FC1_COLW else (H, Y, Ye8)
  input_bufs = (Bias, A, B, scale_A, scale_B, abi_extra[0], abi_extra[1], abi_extra[2],
                *(abi_extra[-2:] if len(abi_extra) == base_extra + 2 else ()),
                *((extra[-1],) if has_expert_counts else ()))
  accesses = tuple(x.index(zero).store(UOp.const(0, x.dtype)) for x in output_bufs) + \
    tuple(x.index(zero).load() for x in input_bufs)
  sink = UOp.sink(*sink_inputs, *accesses,
                  arg=KernelInfo("hk_grouped_mxfp8_gemm_swiglu"+(f"_rk{gptoss_real_k}" if gptoss_real_k != K else "")+
                                 ("_ph1" if gptoss_skip_pad_half else "")+
                                 ("_dsx1" if gptoss_ds_swizzle_xor1 else "")+
                                 ("_ifast1" if gptoss_interior_fast else "")+
                                 ("_soff1" if gptoss_precomputed_soff else "")+
                                 ("_blds1" if gptoss_bias_lds else "")+
                                 ("_cws2" if gptoss_colw_adj_scale2 else "")+
                                 ("_rs4" if gptoss_row_scale_lds4 else "")+
                                 ("_map8c64" if gptoss_map_x8c64 else "_map4c48" if gptoss_map_x4c48 else
                                  "_map2c96" if gptoss_map_x2c96 else "")+
                                 ("_pus1" if gptoss_pad_upper_skip else "")+
                                 ("_oplo1" if gptoss_omit_pad_lower_outputs else "")+
                                 (f"_r72{int(fwd_rocm72)}" if fwd_rocm72_requested else "")+f"_{E}_{M}_{N}_{K}",
                                 estimates=Estimates(ops=2*M*(N-128 if half_n_tail else N)*executed_k,
                                                     mem=(M*K+E*N*K)*A.dtype.itemsize+M*inter*Y.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens_path/"grouped_mxfp8_gemm_swiglu.cpp").read_text()
  # NO -ffast-math: tinygrad compiles the reference (gemm bf16 store, bf16 add, fused_swiglu_quantize) with strict
  # IEEE (-O3, no fast-math). The SwiGLU epilogue uses OCML exp2/log2 + _rn intrinsics, and strict IEEE preserves
  # signed zeros (fp8e4m3 distinguishes -0/+0), so the fused output is byte-identical. The gemm is hardware MFMA +
  # bit/int scale packing, unaffected by dropping fast-math.
  # Register SwiGLU epilogue uses shuffles and small-LDS amax instead of a full-tile LDS round-trip.
  reg_ep = ["-DREG_EPILOGUE=1"] if getenv("REG_EPILOGUE", 1) else []
  # Columnwise quantization reuses dead B MMA buffers for its transpose tile; requires the register epilogue.
  colw = ["-DFUSED_FC1_COLW=1"] if FUSED_FC1_COLW else []
  # Double-buffer the K loop. LOW_LDS selects the single-buffer loop when both flags are set.
  dbuf = ["-DDOUBLE_BUFFER=1"] if getenv("DBUF", 1) else []
  # Single-buffered tile LDS, with the register epilogue.
  low_lds = ["-DLOW_LDS=1"] if getenv("LOW_LDS", 0) else []
  # SCALE_PACK (gated): read raw e8 scales row-major in the scale-load, dropping the mx_pack transpose kernels.
  scale_pack = ["-DSCALE_PACK=1"] if SCALE_PACK else []
  # Retile FC1 to 128x256 independently of the packed/raw scale layout.
  tile_m128 = ["-DTILE_M128"] if TILE_M128 else []
  # Split each gate/linear lane pair across its four accumulator rows so both lanes do useful SwiGLU work.
  split_rows = ["-DFC1_SPLIT_ROWS=1"] if getenv("FC1_SPLIT_ROWS", 1) else []
  native_row = ["-DFC1_NATIVE_ROW=1"] if getenv("FC1_NATIVE_ROW", 1) and NATIVE_MXFP8_CVT else []
  native_cvt = [f"-DNATIVE_MXFP8_CVT={NATIVE_MXFP8_CVT}"] if colw or native_row else []
  xcol_anchor = [f"-DFC1_XCOL_ANCHOR={int(bool(xanchors))}"]
  lib = HIPCCCompiler("gfx950", [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4",
                    "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}",
                    f"-DFC1_OUT_INTER={out_inter}", f"-DFC1_REAL_N={2*real_inter}",
                    f"-DFC1_BIAS_N={bias_n}",
                    f"-DGEMM_E={E}", f"-DMOE_SKIP_EMPTY={getenv('MOE_SKIP_EMPTY', 0)}",
                    f"-DMOE_SKIP_NO_ZERO={getenv('MOE_SKIP_NO_ZERO', 0)}", f"-DGROUPED_WGM={grouped_wgm}",
                    f"-DFC1_FAST_EXP={getenv('FC1_FAST_EXP', 0)}", f"-DFC1_FAST_MATH={getenv('FC1_FAST_MATH', 0)}",
                    f"-DFC1_FAST_RCP={getenv('FC1_FAST_RCP', 0)}",
                    f"-DFC1_FAST_UNROUNDED={getenv('FC1_FAST_UNROUNDED', 0)}",
                    f"-DFC1_HALF_N_TAIL={int(half_n_tail)}",
                    # Two amax scratch banks remove four epilogue barriers per workgroup.
                    f"-DFC1_AMAX_PINGPONG={getenv('FC1_AMAX_PINGPONG', 1)}",
                    # Exact GPT-OSS FC1: 4x2 warps give each warp one complete 32-inter scale block,
                    # deleting cross-warp amax LDS/barriers. Keep generic test shapes on the 2x4 layout.
                    f"-DFC1_WARP_SCALE32={int(gptoss_warp_scale32)}",
                    f"-DFC1_DUAL_COLW_LDS={int(FUSED_FC1_COLW and gptoss_warp_scale32 and bool(getenv('FC1_DUAL_COLW_LDS', 1)))}",
                    f"-DFC1_COLW_ADJ_SCALE2={int(gptoss_colw_adj_scale2)}",
                    f"-DFC1_COLW_PACKED={int(colw_packed)}",
                    f"-DFC1_ROW_SCALE_LDS4={int(gptoss_row_scale_lds4)}",
                    # Avoid LLVM's 16KB LDS materialization of the runtime-indexed A/B buffer bases.
                    f"-DFC1_SCALAR_LDS_BASES={getenv('FC1_SCALAR_LDS_BASES', 1)}",
                    f"-DFC1_FULL_LDS_TILES={int(gptoss_full_lds)}",
                    f"-DFC1_FULL_LDS_TAIL_HALF={int(gptoss_full_lds and half_n_tail and bool(getenv('FC1_FULL_LDS_TAIL_HALF', 1)))}",
                    f"-DFC1_FIXED_WGM_MAP={int(gptoss_fixed_wgm)}",
                    f"-DFC1_VECTOR_TAIL_ZERO={int(gptoss_vector_tail_zero)}",
                    f"-DFC1_SKIP_H_TAIL_ZERO={int(gptoss_skip_h_tail_zero)}",
                    f"-DFC1_VECTOR_EMPTY_ZERO={int(gptoss_vector_empty_zero)}",
                    f"-DFC1_ZERO_NT={int(gptoss_zero_nt)}",
                    f"-DFC1_MED3_CLAMP={int(gptoss_med3)}",
                    f"-DFC1_BINARY_EXPERT={int(gptoss_binary_expert)}",
                    f"-DFC1_DS_SWIZZLE_XOR1={int(gptoss_ds_swizzle_xor1)}",
                    f"-DFC1_REAL_K={gptoss_real_k}",
                    f"-DFC1_PEEL_FINAL={int(gptoss_peel_final)}",
                    f"-DFC1_INTERIOR_FAST={int(gptoss_interior_fast)}",
                    f"-DFC1_PAD_UPPER_SKIP={int(gptoss_pad_upper_skip)}",
                    f"-DFC1_OMIT_PAD_LOWER_OUTPUTS={int(gptoss_omit_pad_lower_outputs)}",
                    f"-DFC1_PRECOMPUTED_SOFF={int(gptoss_precomputed_soff)}",
                    f"-DFC1_BIAS_LDS={int(gptoss_bias_lds)}",
                    f"-DFC1_MAP_XCDS={8 if gptoss_map_x8c64 else 4 if gptoss_map_x4c48 else 2 if gptoss_map_x2c96 else 8}",
                    f"-DFC1_MAP_CHUNK={64 if gptoss_map_x8c64 else 48 if gptoss_map_x4c48 else 96 if gptoss_map_x2c96 else grouped_wgm*grouped_wgm}",
                    f"-DFC1_SKIP_PAD_HALF_ROWS={int(gptoss_skip_pad_half)}",
                    f"-DFC1_PAD_LOWER_REUSE={int(gptoss_pad_lower_reuse)}",
                    f"-DFC1_RESTRICT_READS={getenv('FC1_RESTRICT_READS', 0)}",
                    # Exact 4x2-warp FC1: A has only two 16-row register tiles, so its MFMA consumes
                    # only two packed scale bytes. The generic four-byte helper reads two dead rows.
                    f"-DFC1_PACK_A32={int(gptoss_warp_scale32 and bool(getenv('FC1_PACK_A32', 1)))}",
                    # Reuse each bias value across the two accumulator row tiles and shorten H temps.
                    f"-DFC1_EP_JJ_OUTER={int(gptoss_warp_scale32 and bool(getenv('FC1_EP_JJ_OUTER', 1)))}",
                    f"-DFC1_SKIP_EMPTY_MAIN_ZERO={int(gptoss_skip_empty_main)}",
                    *reg_ep, *dbuf, *low_lds,
                    *scale_pack, *tile_m128, *split_rows, *native_row, *native_cvt, *xcol_anchor, *colw],
                    hipcc_path=fwd_rocm72_hipcc,
                    rocm_path=fwd_rocm72_root).compile_cached(src)
  return hip_program(sink, src, lib)

def grouped_mx_gemm_swiglu(x:Tensor|tuple[Tensor, ...], w:Tensor|tuple[Tensor, ...], expert_off:Tensor,
                           gu_bias:Tensor, real_inter:int|None=None, out_h:Tensor|None=None,
                           padded_inter:int|None=None, expert_counts:Tensor|None=None) -> tuple[Tensor, Tensor, Tensor]:
  # fused FC1 (gate_up gemm + per-expert bias + gpt-oss clamped SwiGLU + mxfp8 quantize). Input prep is byte-
  # identical to grouped_mx_gemm so both paths feed the same quantized operands. gu_bias is the per-expert
  # (E, 2*inter) bf16 bias (padded with zeros to (E, N) here). real_inter (defaults to N//2) trims the padded
  # gate_up columns off y/y_e8. Also returns the FULL-width h (M, N) = pre-swiglu gate_up+bias (bf16), which the
  # model saves so the backward reads it instead of re-running the gemm. Multi-device: outputs shard on the row
  # axis like x. Returns (h (M, N) bf16, y_fp8 (M, real_inter) fp8e4m3, y_e8 (M, real_inter//32) uint8).
  caller_h = out_h
  prepacked_w_si = None
  if (pre_quantized := isinstance(w, tuple)):
    if len(w) == 5:                                        # PRESTORE_WT plus persistent packed forward scales
      w_q, w_e8, _wT_q, _wT_e8, prepacked_w_si = w
      _dgrad_wT_mailbox[w_q.uop] = (_wT_q, _wT_e8)
    elif len(w) == 4:                                      # PRESTORE_WT: 4-tuple (w_q,w_e8,wT_q,wT_e8); register W^T for the dgrad
      w_q, w_e8, _wT_q, _wT_e8 = w
      _dgrad_wT_mailbox[w_q.uop] = (_wT_q, _wT_e8)
    elif len(w) == 3:
      w_q, w_e8, prepacked_w_si = w
    else:
      assert len(w) == 2, f"unsupported prequantized FC1 weight tuple of length {len(w)}"
      w_q, w_e8 = w
    E, N, K2 = w_q.shape
  else:
    E, N, K2 = w.shape
  x_colw, packed_x_si = None, False
  if isinstance(x, tuple):
    assert len(x) in (2, 4, 5), f"unsupported prequantized FC1 activation tuple of length {len(x)}"
    xq_raw, xe_raw = x[:2]
    if len(x) >= 4: x_colw = (x[2], x[3])
    xref, M, Kin = xq_raw, *xq_raw.shape
    x_q = xq_raw.pad(((0, 0), (0, K2 - Kin))) if K2 != Kin else xq_raw
    packed_x_si = xe_raw.dtype == dtypes.uint32
    if packed_x_si:
      assert not SCALE_PACK and Kin == K2 and (E, N, K2) == (32, 5888, 3072) and M % 73728 == 0, \
        "direct packed row_si is restricted to exact GPT-OSS FC1"
      assert xe_raw.shape == (K2 // 128, M), f"bad packed row_si {xe_raw.shape}, expected {(K2 // 128, M)}"
      x_si, xe_in = xe_raw, xe_raw
    else:
      x_e8 = xe_raw.pad(((0, 0), (0, K2 // 32 - Kin // 32)), value=127) if K2 != Kin else xe_raw
    K = K2
  else:
    xref, (M, K) = x, x.shape
    x_q, x_e8, _ = quantize_mxfp8(x)
  # SCALE_PACK: skip the mx_pack transpose; the kernel reads xe_in (raw e8) directly for scales. x_si is then an
  # unused (aliased) scale_A operand of the same buffer size.
  if not packed_x_si:
    x_si = x_e8.reshape(M, K // 32) if SCALE_PACK else mx_pack(x_e8)
    xe_in = x_e8.reshape(M, K // 32)
  assert K == K2, f"shape mismatch K {K} != {K2}"
  assert M % (128 if TILE_M128 else 256) == 0 and N % 256 == 0 and K % 128 == 0, \
    f"grouped mxfp8 swiglu needs M%{128 if TILE_M128 else 256},N%256,K%128, got {M,K}"
  inter = N // 2
  if real_inter is None: real_inter = inter
  assert real_inter <= inter and real_inter % 32 == 0, f"real_inter {real_inter} must be <= {inter} and a mult of 32"
  out_inter = inter if padded_inter is None else padded_inter
  assert out_inter >= inter and out_inter % 32 == 0, f"padded_inter {out_inter} must be >= {inter} and a mult of 32"
  dname = (xref.device[0] if isinstance(xref.device, tuple) else xref.device).split(":")[0]
  use_expert_counts = expert_counts is not None and bool(getenv("FC1_SKIP_PAD_HALF_ROWS", 1)) \
    and (E, N, K) == (32, 5888, 3072) and M % 73728 == 0
  if use_expert_counts:
    assert expert_counts is not None and expert_counts.dtype == dtypes.int32 and expert_counts.shape[-1] == E, \
      f"expert counts must end in ({E},) int32, got {expert_counts.shape} {expert_counts.dtype}"
    # Counts can arrive as a lazy integer REDUCE. Materialize one shared buffer for FC1 and its backward consumers,
    # and mutate the caller-owned Routing tensor so run_layer saves that same substitution-stable buffer identity.
    expert_counts.replace(expert_counts.clone())
  if not pre_quantized: w_q, w_e8, _ = quantize_mxfp8(w)
  w_si = w_e8 if SCALE_PACK else prepacked_w_si if prepacked_w_si is not None else mx_pack_3d(w_e8)
  # Exact GPT-OSS half-tail only computes the 2*real_inter bias columns, so it can read the original compact bias
  # directly. Generic/full-tile shapes retain the padded ABI.
  assert gu_bias.shape[0] == E and gu_bias.dtype == dtypes.bfloat16, f"bias must be (E, *) bf16, got {gu_bias.shape} {gu_bias.dtype}"
  direct_bias = bool(getenv("FC1_DIRECT_BIAS", 1)) and (E, M, N, K) == (32, 73728, 5888, 3072) \
    and bool(getenv("FC1_HALF_N_TAIL", 0)) and gu_bias.shape[1] == 2 * real_inter
  bias = gu_bias.contiguous() if direct_bias else \
    (gu_bias.pad(((0, 0), (0, N - gu_bias.shape[1]))) if gu_bias.shape[1] != N else gu_bias)
  # h is full-width (M, N); y/y_e8 are full-inter-wide (kernel writes N//2), sliced to real_inter after. Shard on
  # the row axis under DP.
  if isinstance(xref.device, tuple) and (row_axis := xref.uop.axis) is not None:
    ndev = len(xref.device)
    def mk(cols, dt): return alloc_like((M, cols), dt, xref.device, row_axis)
    out_h, out_y, out_e8 = mk(N, dtypes.bfloat16), mk(out_inter, FP8_DTYPE), mk(out_inter // 32, dtypes.uint8)
    # columnwise (transposed) outputs (INTER, M*): the M axis is index 1, sharded like x's row axis (transpose of row_axis).
    cax = 1 if row_axis == 0 else 0
    def mkT(mcols, dt):
      return Tensor(Tensor.invalids(*(mcols // ndev if i == cax else s for i, s in enumerate((out_inter, mcols))),
                                    dtype=dt, device=xref.device).uop.unshard(cax), device=xref.device)
    out_yc, out_yc_e8 = (mkT(M, FP8_DTYPE), mkT(M // 32, dtypes.uint8)) if FUSED_FC1_COLW else (None, None)
  else:
    out_h = Tensor.invalids(M, N, dtype=dtypes.bfloat16, device=xref.device)
    out_y = Tensor.invalids(M, out_inter, dtype=FP8_DTYPE, device=xref.device)
    out_e8 = Tensor.invalids(M, out_inter // 32, dtype=dtypes.uint8, device=xref.device)
    out_yc = Tensor.invalids(out_inter, M, dtype=FP8_DTYPE, device=xref.device) if FUSED_FC1_COLW else None
    out_yc_e8 = Tensor.invalids(out_inter, M // 32, dtype=dtypes.uint8, device=xref.device) if FUSED_FC1_COLW else None
  # Write column scales in the exact uint32 layout consumed by FC2 wgrad, avoiding a separate mx_pack.
  if FUSED_FC1_COLW and getenv("GPTOSS_FC1_COLW_PACKED", 0) and (E, N, K, out_inter) == (32, 5888, 3072, 3072) \
      and M == 73728 * (len(xref.device) if isinstance(xref.device, tuple) else 1) \
      and padded_inter == 3072 and (not isinstance(xref.device, tuple) or xref.uop.axis == 0):
    out_yc_e8 = alloc_like((M // 128, out_inter), dtypes.uint32, xref.device, 0 if isinstance(xref.device, tuple) else None)
  if caller_h is not None:
    assert caller_h.shape == (M, N) and caller_h.dtype == dtypes.bfloat16 and caller_h.device == xref.device
    out_h = caller_h
  if x_colw is not None:
    assert x_colw[0].shape == (K, M) and x_colw[0].dtype == FP8_DTYPE, f"bad dispatch q_col {x_colw[0].shape}"
    assert x_colw[1].shape == (M // 128, K) and x_colw[1].dtype == dtypes.uint32, f"bad dispatch si_col {x_colw[1].shape}"
  if packed_x_si:
    assert x_colw is not None, "packed dispatch row_si requires q_col/si_col so FC1 wgrad never needs raw e8"
  x_colw_in = x_colw if x_colw is not None else ()
  colw_in = (out_yc, out_yc_e8) if FUSED_FC1_COLW else ()
  counts_in = (expert_counts,) if use_expert_counts else ()
  # Mark only fresh destinations; a caller-provided checkpoint must keep its storage identity.
  if caller_h is None: out_h = out_h.contiguous()
  if not getenv("DIRECT_FC1_H_ONLY_OUTPUT", 0):
    out_y, out_e8 = out_y.contiguous(), out_e8.contiguous()
    if FUSED_FC1_COLW:
      assert out_yc is not None and out_yc_e8 is not None
      out_yc, out_yc_e8 = out_yc.contiguous(), out_yc_e8.contiguous()
      colw_in = (out_yc, out_yc_e8)
  ys = Tensor.custom_kernel(out_h, out_y, out_e8, bias, x_q, w_q, x_si, w_si, xe_in, w_e8, expert_off, *colw_in, *x_colw_in, *counts_in,
                            fxn=functools.partial(custom_hk_grouped_mxfp8_gemm_swiglu, dname=dname, n_experts=E, real_inter=real_inter,
                                                  has_expert_counts=use_expert_counts),
                            grad_fxn=kernel_grad(_fc1_swiglu_bwd, handoff=(real_inter, use_expert_counts, bool(FUSED_FC1_COLW))))
  # h and y_e8 are returned FULL (unsliced) so the model can SAVE them and the backward's grad_fxn reads those exact
  # (@function-substitutable) UOps -> no fused-fwd re-run and no y_e8 recompute. y_fp8 is sliced to real_inter (the
  # down-gemm's real width); the caller slices y_e8 to real_inter//32 for the down-gemm.
  # FUSED_FC1_COLW: also return the columnwise y (INTER, M) fp8 + e8m0, sliced to real_inter rows (matches the down
  # gemm's activation width after its zero-pad to K2). The caller SAVEs these for the down wgrad (byte-exact with
  # transpose_quantize_mxfp8(x_phys), x_phys = dequant(rowwise y)).
  if FUSED_FC1_COLW:
    if padded_inter is not None: return ys[0], ys[1], ys[2], ys[11], ys[12]
    return ys[0], ys[1][:, :real_inter], ys[2], ys[11][:real_inter], ys[12][:real_inter]
  return (ys[0], ys[1], ys[2]) if padded_inter is not None else (ys[0], ys[1][:, :real_inter], ys[2])

@functools.cache
def custom_hk_grouped_mxfp8_gemm_dswiglu(Dhbf16:UOp, Dhfp8:UOp, Dhe8:UOp, Bias:UOp, GradAq:UOp, Yfwd_e8:UOp,
                                         A:UOp, B:UOp, scale_A:UOp, scale_B:UOp, *extra:UOp, dname:str, n_experts:int) -> UOp:
  # fused FC1 BACKWARD ("Kernel 3"): recompute gate_up = A @ B^T (M, N), fold Bias, apply SwiGLU derivative scaled
  # by GradAq*exp2(127-Yfwd_e8), emit d_h three ways: Dhbf16 (M,N) bf16, Dhfp8 (M,N) fp8e4m3, Dhe8 (M,N//32) e8m0.
  M, K = A.shape
  E, N, K2 = B.shape
  assert K == K2, f"{A.shape} {B.shape}"
  assert E == n_experts, f"{E} != {n_experts}"
  assert Bias.shape == (E, N), f"bias must be (E, N)=({E},{N}), got {Bias.shape}"
  threads = UOp.special(64 * 8, "lidx0")
  workgroups = UOp.special((M // 256) * (N // 256), "gidx0")
  sink_inputs = (Dhbf16.base, Dhfp8.base, Dhe8.base, Bias.base, GradAq.base, Yfwd_e8.base, A.base, B.base,
                 scale_A.base, scale_B.base, extra[0].base, extra[1].base, extra[2].base, threads, workgroups)
  sink = UOp.sink(*sink_inputs,
                  arg=KernelInfo(f"hk_grouped_mxfp8_gemm_dswiglu_{E}_{M}_{N}_{K}",
                                 estimates=Estimates(ops=2*M*N*K, mem=(M*K+E*N*K)*A.dtype.itemsize+M*N*Dhfp8.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens_path/"grouped_mxfp8_gemm_dswiglu.cpp").read_text()
  # NO -ffast-math: strict IEEE (OCML exp2/log2 + _rn intrinsics + signed zeros) to byte-match the reference.
  lib = HIPCCCompiler("gfx950", [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}",
                                 f"-DGEMM_E={E}"]).compile_cached(src)
  return hip_program(sink, src, lib)

def grouped_mx_dswiglu(x:Tensor|tuple[Tensor, Tensor], w:Tensor|tuple[Tensor, Tensor], gu_bias:Tensor, grad_aq:Tensor,
                       y_e8:Tensor, expert_off:Tensor, real_inter:int|None=None) -> tuple[Tensor, Tensor, Tensor]:
  # fused FC1 backward: recompute gate_up (same operands as the fwd), fold per-expert bias, apply the gpt-oss
  # clamped-SwiGLU derivative with incoming grad (grad_aq, cast to bf16 like the reference) scaled by the forward
  # y block-scales (y_e8), and quantize the FULL-width d_h. grad_aq/y_e8 are (M, real_inter)/(M, real_inter//32),
  # zero-padded to the kernel's inter=N//2 so padded gate_up columns produce d_h=0 (sliced off). Returns
  # (dh_bf16 (M, 2*real_inter), dh_fp8 (M, 2*real_inter), dh_e8 (M, 2*real_inter//32)).
  if (pre_quantized := isinstance(w, tuple)):
    if len(w) == 4:                                        # PRESTORE_WT: 4-tuple (w_q,w_e8,wT_q,wT_e8); register W^T for the dgrad
      w_q, w_e8, _wT_q, _wT_e8 = w
      _dgrad_wT_mailbox[w_q.uop] = (_wT_q, _wT_e8)
    else:
      w_q, w_e8 = w
    E, N, K2 = w_q.shape
  else:
    E, N, K2 = w.shape
  if isinstance(x, tuple):
    xq_raw, xe_raw = x
    xref, M, Kin = xq_raw, *xq_raw.shape
    x_q = xq_raw.pad(((0, 0), (0, K2 - Kin))) if K2 != Kin else xq_raw
    x_e8 = xe_raw.pad(((0, 0), (0, K2 // 32 - Kin // 32)), value=127) if K2 != Kin else xe_raw
    K = K2
  else:
    xref, (M, K) = x, x.shape
    x_q, x_e8, _ = quantize_mxfp8(x)
  x_si = mx_pack(x_e8)
  assert K == K2, f"shape mismatch K {K} != {K2}"
  assert M % 256 == 0 and N % 256 == 0 and K % 128 == 0, f"grouped mxfp8 dswiglu needs M%256,N%256,K%128, got {M,K}"
  inter = N // 2
  if real_inter is None: real_inter = inter
  assert real_inter <= inter and real_inter % 32 == 0, f"real_inter {real_inter} must be <= {inter} and a mult of 32"
  dname = (xref.device[0] if isinstance(xref.device, tuple) else xref.device).split(":")[0]
  if not pre_quantized: w_q, w_e8, _ = quantize_mxfp8(w)
  w_si = mx_pack_3d(w_e8)
  xe_in = x_e8.reshape(M, K // 32)
  # per-expert bias (E, 2*real_inter) padded to (E, N); incoming grad cast to bf16 (like the reference) and the
  # grad/scale companions zero-padded on the inter axis to the kernel's full inter=N//2 (padded cols -> d_h=0).
  bias = gu_bias.pad(((0, 0), (0, N - gu_bias.shape[1]))) if gu_bias.shape[1] != N else gu_bias
  grad_aq = grad_aq.cast(dtypes.bfloat16)
  grad_aq = grad_aq.pad(((0, 0), (0, inter - real_inter))) if real_inter != inter else grad_aq
  y_e8 = y_e8.pad(((0, 0), (0, inter // 32 - real_inter // 32))) if real_inter != inter else y_e8
  axis = xref.uop.axis
  dh_bf16 = alloc_like((M, N), dtypes.bfloat16, xref.device, axis)
  dh_fp8 = alloc_like((M, N), FP8_DTYPE, xref.device, axis)
  dh_e8 = alloc_like((M, N // 32), dtypes.uint8, xref.device, axis)
  ys = Tensor.custom_kernel(dh_bf16, dh_fp8, dh_e8, bias, grad_aq, y_e8, x_q, w_q, x_si, w_si, xe_in, w_e8, expert_off,
                            fxn=functools.partial(custom_hk_grouped_mxfp8_gemm_dswiglu, dname=dname, n_experts=E))
  return ys[0][:, :2 * real_inter], ys[1][:, :2 * real_inter], ys[2][:, :2 * real_inter // 32]

@functools.cache
def custom_hk_grouped_mxfp8_gemm_dgrad_dswiglu(Dhbf16:UOp, Dhfp8:UOp, Dhe8:UOp, *args:UOp,
                                                dname:str, n_experts:int, real_inter:int) -> UOp:
  if FUSED_DGRAD_COLW:
    Dhcol, Dhcol_e8, H, A, B, scale_A, scale_B, *extra = args
    colw_outputs = (Dhcol.base, Dhcol_e8.base)
  else:
    H, A, B, scale_A, scale_B, *extra = args
    colw_outputs = ()
  M, K = A.shape
  E, N, K2 = B.shape
  HN = H.shape[1]
  assert K == K2 and H.shape[0] == M, f"{A.shape} {B.shape} {H.shape}"
  assert E == n_experts, f"{E} != {n_experts}"
  assert real_inter <= HN // 2 <= N, f"{real_inter=} H={H.shape} N={N}"
  dbuf_enabled = bool(getenv("DGRAD_DSWIGLU_DBUF", 0))
  packed_io = bool(getenv("DGRAD_PACKED_IO", 0))
  tile_n = 128 if (dbuf_enabled and not packed_io) or getenv("DGRAD_TILE_N128", 0) else 256
  native_row = bool(getenv("DGRAD_NATIVE_ROW", 1) and NATIVE_MXFP8_CVT)
  # Direct-accumulator epilogue is valid only for the packed/native production path; all other paths retain LDS staging.
  direct_ep = bool(getenv("DGRAD_REG_EPILOGUE", 0)) and dbuf_enabled and packed_io and tile_n == 256 and native_row and not FUSED_DGRAD_COLW
  threads = UOp.special(64 * 8, "lidx0")
  workgroups = UOp.special((M // 256) * (N // tile_n), "gidx0")
  sink_inputs = (Dhbf16.base, Dhfp8.base, Dhe8.base, *colw_outputs, H.base, A.base, B.base,
                 scale_A.base, scale_B.base, extra[0].base, extra[1].base, extra[2].base, threads, workgroups)
  sink = UOp.sink(*sink_inputs,
                  arg=KernelInfo(f"hk_grouped_mxfp8_gemm_dgrad_dswiglu_{E}_{M}_{N}_{K}",
                                 estimates=Estimates(ops=2*M*N*K,
                                                     mem=(M*K+E*N*K)*A.dtype.itemsize+M*HN*Dhfp8.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens_path/"grouped_mxfp8_gemm_dgrad_dswiglu.cpp").read_text()
  # The unpacked double-buffered 256x256 tile exhausts the 256-VGPR limit. Packed pair I/O lowers the epilogue
  # footprint enough for the full-width pipeline; otherwise force the validated 256x128 resource-safe tile.
  dbuf = ["-DDOUBLE_BUFFER=1"] if dbuf_enabled else []
  lib = HIPCCCompiler("gfx950", [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}",
                                 f"-DGEMM_E={E}", f"-DH_N={HN}", f"-DREAL_INTER={real_inter}",
                                 f"-DFUSED_DGRAD_COLW={FUSED_DGRAD_COLW}", f"-DNATIVE_MXFP8_CVT={NATIVE_MXFP8_CVT}",
                                 f"-DDGRAD_NATIVE_ROW={int(native_row)}", f"-DDGRAD_REG_EPILOGUE={int(direct_ep)}",
                                 f"-DDGRAD_FAST_EXP={int(bool(getenv('DGRAD_FAST_EXP', 0)) and direct_ep)}",
                                 f"-DDGRAD_PACKED_IO={int(packed_io)}",
                                 f"-DDGRAD_TILE_N128={int(tile_n == 128)}",
                                 f"-DMOE_SKIP_EMPTY={getenv('MOE_SKIP_EMPTY', 0)}", f"-DMOE_SKIP_NO_ZERO={getenv('MOE_SKIP_NO_ZERO', 0)}",
                                 f"-DGROUPED_WGM={getenv('GROUPED_WGM', 8)}", *dbuf]).compile_cached(src)
  return hip_program(sink, src, lib)

def _physical_after_unshard(t:Tensor) -> Tensor:
  # Expose each device's physical shard while retaining all producer CALL barriers above the logical UNSHARD.
  u, barriers = t.uop, []
  while u.op is not Ops.UNSHARD:
    if u.op is Ops.AFTER: barriers.extend(u.src[1:])
    assert u.op in (Ops.AFTER, Ops.RESHAPE, Ops.STAGE), f"cannot localize {u.op}"
    u = u.src[0]
  raw = u.src[0]
  if barriers: raw = raw.after(*barriers)
  return Tensor(raw, device=t.device)

@functools.cache
def custom_hk_grouped_mxfp8_down_dgrad_dual(*args:UOp) -> UOp:
  """GPT-OSS K64 down-dgrad with both dH MXFP8 layouts and ordered dBias partials."""
  m, h, n = 73728, 5888, 3072
  shapes = ((m,h), (h//128,m), (h,m), (m//128,h), (m//32,h), (m,h), (1,33), (1,32),
            (m,n), (32,n,n), (24,m), (32,24,n))
  types = (FP8_DTYPE, dtypes.uint32, FP8_DTYPE, dtypes.uint32, dtypes.float32, dtypes.bfloat16,
           dtypes.int32, dtypes.int32, FP8_DTYPE, FP8_DTYPE, dtypes.uint32, dtypes.uint32)
  assert tuple(a.shape for a in args) == shapes and tuple(a.dtype for a in args) == types
  zero = UOp.const(0, dtypes.int32)
  accesses = tuple(a.index(zero).store(UOp.const(0, a.dtype)) for a in args[:5]) + tuple(a.index(zero).load() for a in args[5:])
  sink = UOp.sink(*(a.base for a in args), *accesses, UOp.special(512, "lidx0"), UOp.special(6912, "gidx0"),
                  arg=KernelInfo("hk_grouped_mxfp8_down_dgrad_dual_packed_gptoss",
                                 estimates=Estimates(ops=2*m*2880*2880+60*m*2880, mem=sum(a.nbytes() for a in args))))
  kittens = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens/"grouped_mxfp8_down_dgrad_dual.cpp").read_text()
  # Retain the qualified derivative rounding and native FP8 conversion: no -ffast-math.
  defines = dict(M_DIM=m, H_DIM=h, DY_STRIDE=n, REAL_INTER=2880, THREADS=256, NATIVE_MXFP8_CVT=1, WRITE_DH=0,
                 GPTOSS_DSWIGLU_RCP_NR=1, GPTOSS_DSWIGLU_SKIP_PAD_TAIL=1, GPTOSS_DSWIGLU_SKIP_EMPTY_TAIL=1,
                 GPTOSS_DSWIGLU_EXPERT_COUNTS=1, GPTOSS_DSWIGLU_XCD_MAP=0, GPTOSS_DSWIGLU_XCD_CHUNK=112,
                 GPTOSS_DSWIGLU_LDS_STRIDE=264, GPTOSS_DSWIGLU_NATIVE_EXP2=1, GPTOSS_DSWIGLU_TAIL_ROW_E8=1)
  lib = HIPCCCompiler("gfx950", [f"-I{kittens/'include'}", "-DKITTENS_CDNA4", "-DHIP_ENABLE_WARP_SYNC_BUILTINS", "-std=c++20",
                                 *[f"-D{k}={v}" for k,v in defines.items()], "-mllvm",
                                 "-amdgpu-sched-strategy=max-memory-clause"]).compile_cached(src)
  return hip_program(sink, src, lib)

def _grouped_mx_down_dgrad_dual(x:tuple[Tensor, Tensor], w:tuple[Tensor, ...], expert_off:Tensor,
                                h:Tensor, real_inter:int, expert_counts:Tensor|None) -> Tensor:
  assert FUSED_DGRAD_DSWIGLU and FUSED_DH_DUAL_PRODUCER and not FUSED_DGRAD_COLW
  assert not getenv("GPTOSS_DSWIGLU_DUAL_WRITE_DH", 0) and not getenv("FUSED_WGRAD_QUANT", 0)
  assert getenv("FUSED_DH_DUAL_COLW", 1) and getenv("FUSED_DH_DUAL_BIAS", 1)
  assert isinstance(x, tuple) and len(x) == 2 and isinstance(w, tuple) and len(w) == 3
  xq, xe = x
  wq, _, wsi = w
  M, N = xq.shape
  devs = xq.device
  ndev = len(devs) if isinstance(devs, tuple) else 1
  assert (M//ndev, N, real_inter) == (73728, 3072, 2880) and wq.shape == (32, 3072, 3072)
  assert h.shape == (M, 5888) and expert_counts is not None
  assert not isinstance(devs, tuple) or (ndev == 8 and xq.uop.axis == h.uop.axis == 0)
  assert xe.shape == (M, 96) and xe.dtype == dtypes.uint8 and wsi.shape == (32, 24, 3072) and wsi.dtype == dtypes.uint32
  assert expert_off.shape == (ndev, 33) and expert_counts.shape == (ndev, 32)
  def mk(shape, dtype, axis): return alloc_like(shape, dtype, devs, axis)
  # Write both scale companions in their GEMM consumers' packed layouts, eliminating two mx_pack passes.
  outputs = (mk((M,5888),FP8_DTYPE,0), mk((46,M),dtypes.uint32,1), mk((5888,M),FP8_DTYPE,1),
             mk((M//128,5888),dtypes.uint32,0), mk((M//32,5888),dtypes.float32,0))
  ys = Tensor.custom_kernel(*outputs, h, expert_off, expert_counts, xq, wq, mx_pack(xe), wsi,
                            fxn=custom_hk_grouped_mxfp8_down_dgrad_dual)
  # Opaque autograd carrier, not numerical dY. _fc1_swiglu_bwd must recover all companions; no fallback is valid.
  # The shape-only cast/slice disappears when the companions are consumed, as checked by full-model graph tests.
  anchor = ys[0].cast(dtypes.bfloat16)
  signal = anchor[:, :3072]
  ent = (anchor.uop, *(y.uop for y in ys[:5]))
  _fused_dswiglu_grad_mailbox[signal.uop] = _fused_dswiglu_grad_mailbox[signal.uop.base] = ent
  return signal

def grouped_mx_gemm_dgrad_dswiglu(x:Tensor|tuple[Tensor, Tensor], w:Tensor|tuple[Tensor, ...], expert_off:Tensor,
                                  h:Tensor, real_inter:int, expert_counts:Tensor|None=None) -> Tensor:
  if getenv("GPTOSS_DOWN_DGRAD_DUAL", 0):
    assert isinstance(x, tuple) and isinstance(w, tuple)
    return _grouped_mx_down_dgrad_dual(x, w, expert_off, h, real_inter, expert_counts)
  prepacked_w_si = w[2] if isinstance(w, tuple) and len(w) == 3 else None
  if (pre_quantized := isinstance(w, tuple)):
    w_q, w_e8 = w[:2]
    E, N, K2 = w_q.shape
  else:
    E, N, K2 = w.shape
  if isinstance(x, tuple):
    xq_raw, xe_raw = x
    xref, M, Kin = xq_raw, *xq_raw.shape
    x_q = xq_raw.pad(((0, 0), (0, K2 - Kin))) if K2 != Kin else xq_raw
    x_e8 = xe_raw.pad(((0, 0), (0, K2 // 32 - Kin // 32)), value=127) if K2 != Kin else xe_raw
    K = K2
  else:
    xref, (M, K) = x, x.shape
    x_q, x_e8, _ = quantize_mxfp8(x)
  assert K == K2, f"shape mismatch K {K} != {K2}"
  assert M % 256 == 0 and N % 256 == 0 and K % 128 == 0, f"fused dgrad dswiglu needs M%256,N%256,K%128, got {M,K}"
  HN = h.shape[1]
  assert h.shape[0] == M and real_inter <= HN // 2 <= N and HN % 32 == 0, f"{h.shape=} {real_inter=} {N=}"
  dname = (xref.device[0] if isinstance(xref.device, tuple) else xref.device).split(":")[0]
  if not pre_quantized: w_q, w_e8, _ = quantize_mxfp8(w)
  x_si = mx_pack(x_e8)
  w_si = prepacked_w_si if prepacked_w_si is not None else mx_pack_3d(w_e8)
  xe_in = x_e8.reshape(M, K // 32)
  row_axis = xref.uop.axis
  dh_bf16 = alloc_like((M, HN), dtypes.bfloat16, xref.device, row_axis)
  dh_fp8 = alloc_like((M, HN), FP8_DTYPE, xref.device, row_axis)
  dh_e8 = alloc_like((M, HN // 32), dtypes.uint8, xref.device, row_axis)
  if FUSED_DGRAD_COLW:
    col_axis = 1-row_axis if row_axis is not None else None
    dh_col = alloc_like((HN, M), FP8_DTYPE, xref.device, col_axis)
    dh_col_e8 = alloc_like((HN, M // 32), dtypes.uint8, xref.device, col_axis)
  outputs = (dh_bf16, dh_fp8, dh_e8, dh_col, dh_col_e8) if FUSED_DGRAD_COLW else (dh_bf16, dh_fp8, dh_e8)
  if getenv("FUSED_DGRAD_DSWIGLU_SPLIT", 0):
    assert not FUSED_DGRAD_COLW, "split dgrad currently emits rowwise dH only"
    from extra.llama_kernels.fused_swiglu_quantize_gptoss import (_custom_swiglu_bwd_fp8_strided,
      _custom_swiglu_bwd_fp8_fast, _custom_swiglu_bwd_fp8_dual)
    # The base grouped GEMM is substantially faster than the epilogue-heavy fused kernel. Its padded BF16 dY is
    # consumed in place by dSwiGLU+quantize, so no shrink/pad/contiguous copy is materialized.
    dy = grouped_mx_gemm((x_q, x_e8), (w_q, w_e8, w_si), expert_off, logical_k=real_inter)
    # DP tensors retain their global logical M here, while the custom callback is lowered once per row shard. Gate
    # the GPT-OSS-only kernel on its physical per-device row count (73,728), not the global DP=8 count (589,824).
    local_M = M // len(xref.device) if isinstance(xref.device, tuple) and xref.uop.axis == 0 else M
    use_dual_dh = bool(FUSED_DH_DUAL_PRODUCER) and (local_M, HN, real_inter, N) == (73728, 5888, 2880, 3072)
    if use_dual_dh:
      # Direct producer: the 32x256 dSwiGLU tile owns both MX block directions. Emit both layouts and 32-row dBias
      # partials; all production consumers use these mailbox companions, so the full BF16 output store is dead.
      assert dname == "AMD", "FUSED_DH_DUAL_PRODUCER is AMD-only"
      assert row_axis in (None, 0), "dual dH producer requires row-sharded grouped activations"
      col_axis = 1 if row_axis == 0 else None
      dh_col = alloc_like((HN, M), FP8_DTYPE, xref.device, col_axis)
      dh_col_e8 = alloc_like((HN, M // 32), dtypes.uint8, xref.device, col_axis)
      dbias_partial = alloc_like((M // 32, HN), dtypes.float32, xref.device, row_axis)
      write_dh = bool(getenv("GPTOSS_DSWIGLU_DUAL_WRITE_DH", 0))
      use_dsw_counts = bool(getenv("GPTOSS_DSWIGLU_EXPERT_COUNTS", 0))
      if use_dsw_counts:
        assert expert_counts is not None and expert_counts.dtype == dtypes.int32 and expert_counts.shape[-1] == E, \
          f"count-aware dSwiGLU requires ({E},) int32 expert counts"
      count_args = (expert_counts,) if use_dsw_counts else ()
      dual_colw, dual_bias = bool(getenv("FUSED_DH_DUAL_COLW", 1)), bool(getenv("FUSED_DH_DUAL_BIAS", 1))
      assert write_dh or (dual_colw and dual_bias), "no-dH dual producer requires both mailbox companions"
      assert write_dh or not getenv("FUSED_WGRAD_QUANT", 0), "no-dH dual producer requires prequantized wgrad"
      if write_dh:
        ys = Tensor.custom_kernel(dh_bf16, dh_fp8, dh_e8, dh_col, dh_col_e8, dbias_partial, h, dy, expert_off, *count_args,
                                  fxn=functools.partial(_custom_swiglu_bwd_fp8_dual, real_inter=real_inter))
        dh_anchor, row_base, signal = ys[0], 1, ys[7]
      else:
        # Do not leave a large unwritten output in the captured graph: even a nominally dead custom-kernel argument
        # participates in buffer planning. The row-q cast is a lazy shape/device anchor and is never materialized
        # while both direct companions are enabled.
        ys = Tensor.custom_kernel(dh_fp8, dh_e8, dh_col, dh_col_e8, dbias_partial, h, dy, expert_off, *count_args,
                                  fxn=functools.partial(_custom_swiglu_bwd_fp8_dual, real_inter=real_inter))
        dh_anchor, row_base, signal = ys[0].cast(dtypes.bfloat16), 0, ys[6]
      # Debug/bring-up gates let the full DP graph fall back independently for either companion while retaining
      # the dual producer. Production consumes both.
      ent = (dh_anchor.uop, ys[row_base].uop, ys[row_base + 1].uop)
      if dual_colw: ent += (ys[row_base + 2].uop, ys[row_base + 3].uop)
      if dual_bias:
        # Preserve the positional mailbox contract: bias partials are slot 5 and therefore require colwise slots.
        assert dual_colw, "dual bias partials require dual colwise mailbox outputs"
        ent += (ys[row_base + 4].uop,)
    else:
      dsw_fxn = _custom_swiglu_bwd_fp8_fast if getenv("DSWIGLU_FAST_MATH", 0) else _custom_swiglu_bwd_fp8_strided
      if getenv("DIRECT_DH_GROUP_OUTPUT", 0): dh_bf16, dh_fp8, dh_e8 = (t.contiguous() for t in (dh_bf16, dh_fp8, dh_e8))
      ys = Tensor.custom_kernel(dh_bf16, dh_fp8, dh_e8, h, dy, fxn=functools.partial(dsw_fxn, real_inter=real_inter))
      ent = tuple(y.uop for y in ys[:3])
      if getenv("FUSED_DH_COLW_SEPARATE", 0):
        # Feed the producer buffer straight to columnwise quantization. Optionally accumulate the 32-row dBias
        # partials from the same BF16 loads, avoiding a full multi-device layout copy and separate reduction.
        local_tq = bool(getenv("DIRECT_DH_LOCAL_TQ", 0))
        tq_in = _physical_after_unshard(ys[0]) if local_tq else ys[0]
        if FUSED_DH_BIAS_PARTIAL:
          from extra.llama_kernels.transpose_quantize_mxfp8 import transpose_quantize_mxfp8_bias
          dh_col, dh_col_e8, _, dbias_partial = transpose_quantize_mxfp8_bias(tq_in)
          if local_tq:
            dh_col = Tensor(dh_col.uop.unshard(1), device=ys[0].device)
            dh_col_e8 = Tensor(dh_col_e8.uop.unshard(1), device=ys[0].device)
            dbias_partial = Tensor(dbias_partial.uop.unshard(0), device=ys[0].device)
          ent += (dh_col.uop, dh_col_e8.uop, dbias_partial.uop)
        else:
          from extra.llama_kernels.transpose_quantize_mxfp8 import transpose_quantize_mxfp8
          dh_col, dh_col_e8, _ = transpose_quantize_mxfp8(tq_in)
          if local_tq:
            dh_col = Tensor(dh_col.uop.unshard(1), device=ys[0].device)
            dh_col_e8 = Tensor(dh_col_e8.uop.unshard(1), device=ys[0].device)
          ent += (dh_col.uop, dh_col_e8.uop)
    # In dual mode the BF16 dH allocation is intentionally unwritten. Carry the valid dY input through the CALL as
    # the autograd signal and attach the companions to that UOp. This preserves the dependency and makes a future
    # mailbox miss safely fall back from the real dY instead of interpreting an invalid allocation as a gradient.
    signal = signal if use_dual_dh else ys[0]
    _fused_dswiglu_grad_mailbox[signal.uop] = _fused_dswiglu_grad_mailbox[signal.uop.base] = ent
    return signal if use_dual_dh else signal[:, :N]

  ys = Tensor.custom_kernel(*outputs, h, x_q, w_q, x_si, w_si, xe_in, w_e8, expert_off,
                            fxn=functools.partial(custom_hk_grouped_mxfp8_gemm_dgrad_dswiglu,
                                                dname=dname, n_experts=E, real_inter=real_inter))
  ent = tuple(y.uop for y in ys[:len(outputs)])
  _fused_dswiglu_grad_mailbox[ys[0].uop] = _fused_dswiglu_grad_mailbox[ys[0].uop.base] = ent
  return ys[0][:, :N]

def _fc1_bias_grad(dh:Tensor, expert_off:Tensor, E:int) -> Tensor:
  # d_bias[e,col] = sum over all rows assigned to expert e of dh[row,col]. Reconstruct the per-256-tile expert id
  # (tile_e) from expert_off (mirrors Routing.tile_e), sum dh over each tile's 256 rows, then contract by the
  # tile_e onehot. The sum over the G (device) axis allreduces the replicated bias grad, matching _moe_bias_tile's
  # backward (tile-onehot^T @ tile-summed d_h). Returns (E, N) fp32; the pad-backward shrinks it to the raw bias.
  M, N = dh.shape
  G = expert_off.shape[0]
  tiles = (M // G) // 256
  dev = expert_off.device
  tr = Tensor.arange(tiles, dtype=dtypes.int32).reshape(1, tiles, 1) * 256
  tr = tr.shard(dev) if isinstance(dev, tuple) else tr.to(dev)
  tile_e = ((tr >= expert_off[:, :E].reshape(G, 1, E)).sum(-1) - 1).cast(dtypes.int32).reshape(-1)   # (G*tiles,)
  tile_sums = dh.reshape(G, tiles, 256, N).float().sum(2).reshape(-1, N)                              # (G*tiles, N) fp32 acc
  # onehot(tile_e)^T @ tile_sums exactly like _moe_bias_tile's backward: contracts over the sharded G*tiles axis,
  # allreducing the replicated bias grad. Byte-matches the current path (broadcast-multiply-sum diverges under DP).
  return tile_e.one_hot(E).float().transpose() @ tile_sums                                           # (E, N)

@functools.cache
def _custom_bias_partials_reduce(out:UOp, partial:UOp, expert_off:UOp, *, dname:str, n_experts:int) -> UOp:
  P, N = partial.shape
  assert out.shape == (1, n_experts, N) and expert_off.shape[-1] == n_experts + 1 and N % 256 == 0
  gptoss_fc1 = (P, N, n_experts) == (2304, 5888, 32)
  gptoss_down = (P, N, n_experts) == (2304, 3072, 32)
  # Reuse the ordered two-accumulator loop for down-bias partials, without changing FC1 or the DP aggregation boundary.
  reduce_unroll = (getenv("GPTOSS_BIAS_REDUCE_UNROLL", 32) if gptoss_fc1 else
                   getenv("GPTOSS_DOWN_BIAS_REDUCE_UNROLL", 1) if gptoss_down else 1)
  assert reduce_unroll in (1, 2, 4, 8, 16, 32, 64)
  threads, groups = UOp.special(256, "lidx0"), UOp.special(n_experts * (N // 256), "gidx0")
  # The executable comes from the precompiled HIP binary, but HCQ still derives cross-queue buffer hazards from
  # LOAD/STORE nodes in this metadata SINK. Without them ProgramInfo.outs/ins are empty, so the following DP
  # all-reduce may let its copy queue read `out` before this compute-queue kernel has finished writing it.
  zero = UOp.const(0, dtypes.int32)
  accesses = (out.index(zero).store(UOp.const(0.0, dtypes.float32)),
              partial.index(zero).load(), expert_off.index(zero).load())
  sink = UOp.sink(out.base, partial.base, expert_off.base, *accesses, threads, groups,
                  arg=KernelInfo(f"bias_partials_reduce_{P}_{N}_{n_experts}" +
                                 (f"_u{reduce_unroll}" if reduce_unroll != 1 else ""),
                                 estimates=Estimates(ops=P*N, mem=(P+n_experts)*N*4)))
  src = (pathlib.Path(__file__).parent.parent/"llama_kernels"/"transpose_quantize_mxfp8"/"bias_partials_reduce.cpp").read_text()
  flags = ["-std=c++20", "-ffast-math", f"-DPARTIAL_ROWS={P}", f"-DN_COLS={N}", f"-DN_EXPERTS={n_experts}",
           f"-DREDUCE_UNROLL={reduce_unroll}"]
  lib = HIPCCCompiler("gfx950", flags).compile_cached(src)
  return hip_program(sink, src, lib)

def _fc1_bias_grad_partials_hip(tile_sums:Tensor, expert_off:Tensor, E:int) -> Tensor:
  _, N = tile_sums.shape
  is_multi = isinstance(tile_sums.device, tuple)
  inv = Tensor.invalids(1, E, N, dtype=dtypes.float32, device=tile_sums.device)
  out = Tensor(inv.uop.unshard(0), device=tile_sums.device) if is_multi else inv
  out, *_ = Tensor.custom_kernel(out, tile_sums, expert_off,
    fxn=functools.partial(_custom_bias_partials_reduce, dname="AMD", n_experts=E))
  return out.sum(0) if is_multi else out.squeeze(0)

def _fc1_bias_grad_partials(tile_sums:Tensor, expert_off:Tensor, E:int, tile_rows:int=32) -> Tensor:
  # Direct dH producer already accumulated each 256-row tile in FP32. Contract those compact partials by expert
  # exactly like _fc1_bias_grad, without ever materializing the full BF16 dH tensor.
  P, N = tile_sums.shape
  G, dev = expert_off.shape[0], expert_off.device
  assert P % G == 0
  tiles = P // G
  tr = Tensor.arange(tiles, dtype=dtypes.int32).reshape(1, tiles, 1) * tile_rows
  tr = tr.shard(dev) if isinstance(dev, tuple) else tr.to(dev)
  tile_e = ((tr >= expert_off[:, :E].reshape(G, 1, E)).sum(-1) - 1).cast(dtypes.int32).reshape(-1)
  return tile_e.one_hot(E).float().transpose() @ tile_sums.reshape(-1, N)

def _fc1_swiglu_bwd(gradient:UOp, kernel:UOp, *, handoff:tuple[int, bool, bool]) -> tuple:
  if getenv("GPTOSS_DOWN_DGRAD_DUAL", 0):
    assert _fused_dswiglu_lookup(gradient) is not None, "fused dH companions lost: numerical fallback is forbidden"
  # backward for grouped_mx_gemm_swiglu (Option C: NO gemm recompute). Read the SAVED h (a fwd kernel output that
  # the model checkpoints) and run the existing _custom_swiglu_bwd_fp8 to get d_h from (grad_aq, y_e8, h) — exactly
  # like the FUSED_SWIGLU backward. Then dgrad (dh_fp8 tuple path) + wgrad (dh_bf16) + d_bias, mirroring
  # custom_grouped_mx_gemm_bw. grad_aq is the down-gemm's grad w.r.t. y_fp8 (= physical grad * block_scale(y_e8));
  # _custom_swiglu_bwd_fp8 multiplies by exp2(127-y_e8)=1/scale so they cancel to the physical grad.
  from extra.llama_kernels.fused_swiglu_quantize_gptoss import _custom_swiglu_bwd_fp8
  inputs = kernel.src[1:]   # (H, out_y, out_e8, bias, x_q, w_q, x_si, w_si, xe_in, w_e8, expert_off, [y_col,y_e8])
  aq = Tensor(inputs[4], device=inputs[4].device)
  bq = Tensor(inputs[5], device=inputs[5].device)
  ae8 = Tensor(inputs[8], device=inputs[8].device)
  be8 = Tensor(inputs[9], device=inputs[9].device)
  expert_off = Tensor(inputs[10], device=inputs[10].device)
  E, N = bq.shape[0], bq.shape[1]
  M, K = aq.shape
  inter = N // 2
  # .after(kernel): inputs[0]/inputs[2] are the fwd kernel's H and y_e8 OUTPUT buffers; the raw ref is pre-write
  # (invalids) -> attach the kernel-write barrier (feedback_custom_kernel_call_src). Both are model-SAVED (full,
  # unsliced), so the same UOp is @function-substituted -> read the checkpoint, no fused-fwd re-run, no recompute.
  h = Tensor(inputs[0].after(kernel), device=aq.device).reshape(M, N)                     # SAVED h (gate_up+bias)
  dh_colw, dbias_partial = None, None
  if (fused_dh := _fused_dswiglu_lookup(gradient)) is not None:
    dh_bf16, dh_fp8, dh_e8 = (Tensor(u, device=aq.device) for u in fused_dh[:3])
    if len(fused_dh) >= 5:
      col_scale = Tensor(fused_dh[4], device=aq.device)
      dh_colw = (Tensor(fused_dh[3], device=aq.device), col_scale if col_scale.dtype == dtypes.uint32 else mx_pack(col_scale))
    if len(fused_dh) == 6: dbias_partial = Tensor(fused_dh[5], device=aq.device)
  else:
    # A gpt-oss-specific padded FC1 output can be wider than the computed N//2 columns. Its zero tail is consumed
    # directly by FC2, but dSwiGLU still operates on the original computed width.
    out_inter = inputs[1].shape[1]
    y_e8 = Tensor(inputs[2].after(kernel), device=aq.device)[:, :inter // 32]              # SAVED y_e8 (fwd block-scales)
    grad_aq = Tensor(gradient, device=aq.device).reshape(M, out_inter)[:, :inter].cast(dtypes.bfloat16)
    axis = h.uop.axis if isinstance(h.device, tuple) else None
    gh = alloc_like((M, N), dtypes.bfloat16, h.device, axis)
    gfp8 = alloc_like((M, N), FP8_DTYPE, h.device, axis)
    ge8 = alloc_like((M, N // 32), dtypes.uint8, h.device, axis)
    dh_bf16, dh_fp8, dh_e8, *_ = Tensor.custom_kernel(gh, gfp8, ge8, h, grad_aq, y_e8, fxn=_custom_swiglu_bwd_fp8)
  w_op = _dgrad_wT_operand(inputs[5]) or _dynamic_dgrad_wT_operand(bq, be8, Tensor(inputs[7], device=aq.device))
  # Restore the original FC1 activation's rowwise block scale in the grouped-GEMM epilogue. This retains the old
  # bf16 GEMM-store then bf16 multiply-store rounding contract while avoiding a full grad_x HBM read/write pass.
  grad_x = grouped_mx_gemm((dh_fp8, dh_e8), w_op, expert_off, out_scale=ae8)
  # Optional caller-owned dispatch columnwise buffers are dependency-only FC1 inputs. Keeping them explicit makes
  # them available to wgrad without reconstructing the dispatch CALL or copying the full buffers at run_layer exit.
  xanchor_off = 13 if FUSED_FC1_COLW else 11
  x_colw = (Tensor(inputs[xanchor_off], device=aq.device), Tensor(inputs[xanchor_off + 1], device=aq.device)) \
    if len(inputs) >= xanchor_off + 2 and inputs[xanchor_off + 1].dtype == dtypes.uint32 else None
  # The exact routing counts are already a saved input of the GPT-OSS FC1 forward custom call. Reuse that local
  # per-device buffer in wgrad so its contraction stops before a wholly padded final 128-row tile.
  expert_counts = Tensor(inputs[-1], device=aq.device) \
    if inputs[-1].dtype == dtypes.int32 and inputs[-1].shape[-1] == E else None
  wb = grouped_mx_wgrad(dh_bf16, None, expert_off, E, g_colw=dh_colw,
                        xg_colw=x_colw, xg_rowwise=None if x_colw is not None else (aq, ae8),
                        # A direct dual producer already supplied exact bias partials. Do not select the alternate
                        # fused-wgrad dBias path, since that path would be the sole reader of its intentionally dead
                        # BF16 dH output slot.
                        bias_grad=bool(getenv("FUSED_WGRAD_DBIAS", 0)) and dbias_partial is None,
                        expert_counts=expert_counts)
  grad_w, d_bias = wb if isinstance(wb, tuple) else (
    wb, (_fc1_bias_grad_partials_hip(dbias_partial, expert_off, E) if getenv("FUSED_DH_BIAS_REDUCE_HIP", 0)
         else _fc1_bias_grad_partials(dbias_partial, expert_off, E)) if dbias_partial is not None else _fc1_bias_grad(dh_bf16, expert_off, E))
  # FC1_DIRECT_BIAS passes the model's compact (E, 2*real_inter) bias directly to the fused forward while H/dH
  # retain the physical padded N columns. Autograd must return the compact input shape, not the physical dH width.
  d_bias = d_bias[:, :inputs[3].shape[1]]
  grad_xq = grad_x
  grad_wq = _finish_expert_wgrad(grad_w)                                                  # w_stored (gate_up weight is fp8)
  # grads for base inputs plus optional out_yc/out_yc_e8 outputs.
  # FUSED_FC1_COLW appends 2 pure-output columnwise buffers (no grad flows through them) -> trailing None slots.
  return (None, None, None, d_bias.uop, grad_xq.uop, grad_wq.uop, None, None, None, None, None) + (None,) * (len(inputs) - 11)

@functools.cache
def custom_hk_grouped_mxfp8_wgrad(C:UOp, A:UOp, B:UOp, scale_A:UOp, scale_B:UOp, expert_off:UOp, *extra:UOp,
                                  dname:str, n_experts:int, direct_shards:bool=False) -> UOp:
  N, M = A.shape
  K, M2 = B.shape
  assert M == M2, f"{A.shape} {B.shape}"
  E = n_experts
  shard_outs, extra = (extra[-7:], extra[:-7]) if direct_shards else ((), extra)
  has_expert_counts = bool(extra) and extra[-1].dtype == dtypes.int32 and extra[-1].shape[-1] == E
  abi_extra = extra[:-1] if has_expert_counts else extra
  fused_bias = len(abi_extra) == 2
  gptoss_fc1_tail = bool(getenv("GPTOSS_WGRAD_TAIL", 0)) and (E, M, N, K) == (32, 73728, 5888, 3072)
  gptoss_down = (E, M, N, K) == (32, 73728, 3072, 3072)
  # The physical down-weight gradient is 3072x3072, but only its leading 2880x2880 is live. In the final output
  # tile, compute only the leading 64x64 quadrant; zero accumulators still overwrite every physical padding element.
  gptoss_down_tail = gptoss_down and not fused_bias and bool(getenv("GPTOSS_DOWN_WGRAD_TAIL", 1))
  # The runtime-indexed ping/pong branch intermittently consumes the wrong LDS plane for DOWN. Use the literal
  # stage-0/stage-1 pair loop already proven by the FC1 tail specialization while retaining global/LDS overlap.
  pair_loop = bool(getenv("GPTOSS_WGRAD_PAIR_LOOP", 1)) and (gptoss_fc1_tail or gptoss_down)
  # The exact non-fused FC1-tail grid has 242/276 interior output tiles. Dispatch those tiles once to a hot loop
  # without repeated N/K tail predicates; the edge tiles retain the guarded loop. The analogous down path has
  # 121/144 interior tiles and uses a separate opt-out after exact two-GPU qualification.
  interior_fast = (gptoss_fc1_tail and not fused_bias and bool(getenv("GPTOSS_WGRAD_INTERIOR_FAST", 1))) or \
                  (gptoss_down_tail and bool(getenv("GPTOSS_DOWN_WGRAD_INTERIOR_FAST", 1)))
  # Issue the current A operand before both B operands on the exact FC1 path. This leaves MFMA order unchanged,
  # lowers the ROCm 7.2.1 allocation from 236 to 234 VGPR, and is kept off for DOWN/general wgrad.
  a_first = gptoss_fc1_tail and not fused_bias and bool(getenv("GPTOSS_WGRAD_A_FIRST", 1))
  # On the predicate-free FC1 interior path, four B1 LDS reads can finish underneath the first A/B0 MFMA. Keep
  # the library default conservative; the GPT-OSS launch enables this exact-shape schedule after exact ABBA tests.
  b1_overlap = gptoss_fc1_tail and not fused_bias and interior_fast and bool(getenv("GPTOSS_WGRAD_B1_OVERLAP", 0))
  # Fetch the two small scale tiles before the four large operand tiles so their global latency is hidden by the
  # raw-buffer-to-LDS transfers. This is separately gated to the exact FC1 interior path.
  scale_first = gptoss_fc1_tail and not fused_bias and interior_fast and bool(getenv("GPTOSS_WGRAD_SCALE_FIRST", 0))
  # RT_B is 32 rows for every specialization of this GPT-OSS grouped wgrad. The generic 64-row scale packer
  # reads two unused scale pairs past the end of the last warp's 256-row LDS tile; use its exact 32-row packer.
  pack_b32 = bool(getenv("GPTOSS_WGRAD_PACK_B32", 1))
  # Keep the ordinary machine scheduler: after fixing the LDS pointer intrinsic it is exact under both FC1 and
  # DOWN shapes and slightly faster than max-ILP. The scheduler flag remains available for focused comparisons.
  fc1_max_ilp = gptoss_fc1_tail and bool(getenv("GPTOSS_WGRAD_MAX_ILP", 0))
  # Interleave the exact FC1 grid across two XCDs. The submission selects the independently qualified WGM6/chunk128
  # locality mapping; keep the library defaults at the earlier WGM1/chunk96 mapping for diagnostic A/B runs.
  fc1_xcd_map = gptoss_fc1_tail and bool(getenv("GPTOSS_WGRAD_XCD_MAP", 1))
  # The square exact down-wgrad has a different locality optimum: two XCDs in 128-workgroup chunks. Full-buffer
  # signed poison/repeat checks were byte-exact; ABBA64 improved 2.62-3.34% on two MI350X GPUs, resources unchanged.
  down_xcd_map = gptoss_down_tail and bool(getenv("GPTOSS_DOWN_WGRAD_XCD_MAP", 1))
  # Keep grouped-wgrad library defaults conservative. The GPT-OSS launch script selects exact-shape locality maps;
  # these only permute workgroups and leave the MFMA/load/store math unchanged.
  fc1_wgrad_wgm = getenv("GPTOSS_WGRAD_WGM", 1) if gptoss_fc1_tail else 1
  fc1_wgrad_xcd_chunk = getenv("GPTOSS_WGRAD_XCD_CHUNK", 96) if fc1_xcd_map else 96
  down_wgrad_wgm = getenv("GPTOSS_DOWN_WGRAD_WGM", 1) if gptoss_down_tail else 1
  down_wgrad_xcd_chunk = getenv("GPTOSS_DOWN_WGRAD_XCD_CHUNK", 128) if down_xcd_map else 128
  # Each final down-wgrad edge tile has only 64 live rows/columns. Load just that first half of its surviving
  # 128-row A/B LDS tile; the inactive warp_m/warp_n values never consume the omitted half.
  down_edge_load64 = gptoss_down_tail and bool(getenv("GPTOSS_DOWN_WGRAD_EDGE_LOAD64", 1))
  wgrad_xcd_map = fc1_xcd_map or down_xcd_map
  wgrad_xcd_chunk = fc1_wgrad_xcd_chunk if fc1_xcd_map else down_wgrad_xcd_chunk
  # Select the optional ROCm toolchain separately for FC1 and down wgrad; fused-bias FC1 retains the default.
  wgrad_rocm72_requested = (gptoss_fc1_tail and not fused_bias and bool(getenv("GPTOSS_WGRAD_ROCM72", 1))) or \
                           (gptoss_down_tail and bool(getenv("GPTOSS_DOWN_WGRAD_ROCM72", 0)))
  wgrad_rocm72_hipcc, wgrad_rocm72_root = rocm_toolchain("/opt/rocm-7.2.1", wgrad_rocm72_requested,
    flag="GPTOSS_WGRAD_ROCM72")
  wgrad_rocm72 = wgrad_rocm72_root is not None
  assert not abi_extra or (abi_extra[0].shape == (1, E * N) and abi_extra[1].shape == (M, N))
  assert not has_expert_counts or gptoss_fc1_tail or gptoss_down_tail, \
    "expert-count contraction is restricted to exact GPT-OSS FC1/down wgrad"
  if direct_shards:
    assert has_expert_counts and not fused_bias and E == 32 and len(shard_outs) == 7
    assert C.shape == (1, 4*N, K) and all(x.shape == C.shape and x.dtype == dtypes.bfloat16 for x in (C, *shard_outs))
  threads = UOp.special(64 * 8, "lidx0")
  workgroups = UOp.special(E * (N // 256) * (K // 256), "gidx0")
  # Sum of expert contraction lengths is at most M; runtime counts can shorten each loop further.
  # This precompiled GEMM is followed by ZeRO-2 peer copies. Describe its actual accesses so the HCQ graph makes
  # those copy queues wait for the compute queue instead of racing a partially-written gradient buffer.
  zero = UOp.const(0, dtypes.int32)
  out_bufs, in_bufs = ((C, abi_extra[0]), (A, B, scale_A, scale_B, expert_off, abi_extra[1])) if fused_bias else \
                      ((C,), (A, B, scale_A, scale_B, expert_off))
  out_bufs += shard_outs
  if has_expert_counts: in_bufs += (extra[-1],)
  accesses = tuple(x.index(zero).store(UOp.const(0, x.dtype)) for x in out_bufs) + tuple(x.index(zero).load() for x in in_bufs)
  sink = UOp.sink(C.base, A.base, B.base, scale_A.base, scale_B.base, expert_off.base,
                  *(x.base for x in (*extra, *shard_outs)), *accesses, threads, workgroups,
                  arg=KernelInfo(
                    f"hk_{'gptoss_fc1_wgrad_tail' if gptoss_fc1_tail else 'gptoss_down_wgrad_tail' if gptoss_down_tail else 'grouped_mxfp8_wgrad'}"+
                    f"_pair{int(pair_loop)}b32{int(pack_b32)}"+
                    (f"ifast{int(interior_fast)}" if gptoss_fc1_tail or gptoss_down_tail else "")+
                    (f"afirst{int(a_first)}" if gptoss_fc1_tail else "")+
                    (f"b1ov{int(b1_overlap)}" if gptoss_fc1_tail else "")+
                    (f"sf{int(scale_first)}" if gptoss_fc1_tail else "")+
                    (f"ilp{int(fc1_max_ilp)}map{int(fc1_xcd_map)}cnt{int(has_expert_counts)}"
                     f"wgm{fc1_wgrad_wgm}c{fc1_wgrad_xcd_chunk}_r72{int(wgrad_rocm72)}" if gptoss_fc1_tail else
                     f"map{int(down_xcd_map)}e64{int(down_edge_load64)}cnt{int(has_expert_counts)}"
                     f"wgm{down_wgrad_wgm}c{down_wgrad_xcd_chunk}_r72{int(wgrad_rocm72)}" if gptoss_down_tail else "")+
                    f"_ldsptr1_{E}_{M}_{N}_{K}",
                    estimates=Estimates(ops=2*M*(5760 if gptoss_fc1_tail else 2880 if gptoss_down_tail else N)*
                                       (2880 if gptoss_fc1_tail or gptoss_down_tail else K),
                                        mem=(N*M+K*M)*A.dtype.itemsize+E*N*K*C.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  # Native 32x32 MFMA is restricted to this exact FC1 configuration, not FC2.
  native32 = bool(getenv("GPTOSS_FC1_WGRAD_NATIVE32", 0)) and gptoss_fc1_tail and direct_shards and \
    has_expert_counts and not fused_bias and wgrad_rocm72 and pair_loop and pack_b32 and interior_fast and \
    a_first and b1_overlap and scale_first and not fc1_max_ilp and fc1_xcd_map and \
    (fc1_wgrad_wgm, fc1_wgrad_xcd_chunk, getenv("WGRAD_DBUF", 0)) == (6, 128, 1)
  src = (kittens_path/("grouped_mxfp8_wgrad_fc1_native32.cpp" if native32 else "grouped_mxfp8_wgrad.cpp")).read_text()
  sched_flags = ["-mllvm", "-amdgpu-sched-strategy=max-ilp"] if fc1_max_ilp else []
  compiler_flags = [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                    "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DWGRAD_M={M}", f"-DWGRAD_N={N}", f"-DWGRAD_K={K}",
                    f"-DWGRAD_E={E}", f"-DWGRAD_DBUF={getenv('WGRAD_DBUF', 0)}",
                    f"-DGPTOSS_WGRAD_TAIL={int(gptoss_fc1_tail)}",
                    f"-DGPTOSS_DOWN_WGRAD_TAIL={int(gptoss_down_tail)}",
                    f"-DGPTOSS_WGRAD_PAIR_LOOP={int(pair_loop)}",
                    f"-DGPTOSS_WGRAD_PACK_B32={int(pack_b32)}",
                    f"-DGPTOSS_WGRAD_INTERIOR_FAST={int(interior_fast)}",
                    f"-DGPTOSS_WGRAD_A_FIRST={int(a_first)}",
                    f"-DGPTOSS_WGRAD_B1_OVERLAP={int(b1_overlap)}",
                    f"-DGPTOSS_WGRAD_SCALE_FIRST={int(scale_first)}",
                    f"-DGPTOSS_WGRAD_WGM={fc1_wgrad_wgm if gptoss_fc1_tail else down_wgrad_wgm}",
                    f"-DGPTOSS_WGRAD_XCD_MAP={int(wgrad_xcd_map)}",
                    "-DGPTOSS_WGRAD_XCDS=2", f"-DGPTOSS_WGRAD_XCD_CHUNK={wgrad_xcd_chunk}",
                    f"-DGPTOSS_DOWN_WGRAD_EDGE_LOAD64={int(down_edge_load64)}",
                    f"-DGPTOSS_WGRAD_EXPERT_COUNTS={int(has_expert_counts)}",
                    f"-DGPTOSS_WGRAD_SHARDS={int(direct_shards)}",
                    f"-DFUSED_WGRAD_DBIAS={int(fused_bias)}", "-DWGRAD_SAFE_LDS_PTR=1", *sched_flags]
  lib = HIPCCCompiler("gfx950", compiler_flags,
                      hipcc_path=wgrad_rocm72_hipcc,
                      rocm_path=wgrad_rocm72_root).compile_cached(src)
  return hip_program(sink, src, lib)

@functools.cache
def custom_hk_grouped_mxfp8_wgrad_fused(C:UOp, g:UOp, x:UOp, expert_off:UOp, *, dname:str, n_experts:int) -> UOp:
  # FUSED_WGRAD_QUANT: the wgrad gemm reads bf16 g (M,N) and xg (M,K) DIRECTLY and folds the transpose + mxfp8
  # block-quantize into its LDS-load stage (no separate transpose_quantize_mxfp8 kernels, no fp8 HBM intermediates).
  M, N = g.shape
  M2, K = x.shape
  assert M == M2, f"{g.shape} {x.shape}"
  E = n_experts
  threads = UOp.special(64 * 8, "lidx0")
  workgroups = UOp.special(E * (N // 256) * (K // 256), "gidx0")
  sink = UOp.sink(C.base, g.base, x.base, expert_off.base, threads, workgroups,
                  arg=KernelInfo(f"hk_grouped_mxfp8_wgrad_fused_{E}_{M}_{N}_{K}",
                                 estimates=Estimates(ops=2*M*N*K, mem=(N*M+K*M)*g.dtype.itemsize+E*N*K*C.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens_path/"grouped_mxfp8_wgrad_fused.cpp").read_text()
  lib = HIPCCCompiler("gfx950", [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DWGRAD_M={M}", f"-DWGRAD_N={N}", f"-DWGRAD_K={K}",
                                 f"-DWGRAD_E={E}", f"-DWGRAD_DBUF={getenv('WGRAD_DBUF', 1)}",
                                 f"-DNATIVE_MXFP8_CVT={getenv('NATIVE_MXFP8_CVT', 0)}"]).compile_cached(src)
  return hip_program(sink, src, lib)

def grouped_mx_wgrad(g:Tensor, xg:Tensor|None, expert_off:Tensor, n_experts:int,
                     xg_colw:tuple[Tensor, Tensor]|None=None, g_colw:tuple[Tensor, Tensor]|None=None,
                     xg_rowwise:tuple[Tensor, Tensor]|None=None, bias_grad:bool=False,
                     g_direct:bool=False, expert_counts:Tensor|None=None) -> Tensor|tuple[Tensor, Tensor]:
  # xg_colw (FUSED_FC1_COLW): a PRE-EMITTED columnwise mxfp8 of xg = (xT (K, M) fp8, x_si packed e8), produced once
  # in the producing kernel's epilogue (K1) and byte-exact with transpose_quantize_mxfp8(xg). When given, skip the
  # separate transpose_quantize(xg) AND the xg dequant-materialize that feeds it (xg then goes unused). g_colw is the
  # analogous pre-emitted (gT, g_si) from the fused FC2-dgrad+dSwiGLU epilogue.
  from extra.llama_kernels.transpose_quantize_mxfp8 import transpose_quantize_mxfp8
  M, N = g.shape
  assert sum(x is not None for x in (xg, xg_colw, xg_rowwise)) == 1, "provide exactly one wgrad activation form"
  K = xg.shape[1] if xg is not None else xg_colw[0].shape[0] if xg_colw is not None else xg_rowwise[0].shape[1]
  assert xg is None or M == xg.shape[0], f"{g.shape} {xg.shape}"
  assert xg_rowwise is None or xg_rowwise[0].shape == (M, K)
  assert M % 128 == 0 and N % 256 == 0 and K % 256 == 0, f"wgrad needs M%128,N%256,K%256, got {g.shape} K={K}"
  dname = (g.device[0] if isinstance(g.device, tuple) else g.device).split(":")[0]
  is_multi = isinstance(g.device, tuple)
  inv = Tensor.invalids(1, n_experts * N, K, dtype=dtypes.bfloat16, device=g.device)
  out = Tensor(inv.uop.unshard(0), device=g.device) if is_multi else inv
  do_bias = bias_grad and bool(getenv("FUSED_WGRAD_DBIAS", 0))
  do_tq_bias = bias_grad and bool(getenv("FUSED_TQ_WGRAD_DBIAS", 0))
  assert not (do_bias and do_tq_bias) and (not do_tq_bias or g_colw is None)
  # `g.shape` is the logical DP tensor (for DP2, 147456 rows), while each custom-kernel launch sees one 73728-row
  # shard. Select the exact GPT-OSS specialization from that physical shape, matching the callback placeholders.
  local_m = g.uop.shard_shape[0] if is_multi and g.uop.axis is not None else M
  count_shape = (n_experts, local_m, N, K)
  use_expert_counts = expert_counts is not None and bool(getenv("GPTOSS_WGRAD_EXPERT_COUNTS", 1)) and \
    ((count_shape == (32, 73728, 5888, 3072) and bool(getenv("GPTOSS_WGRAD_TAIL", 0))) or
     (count_shape == (32, 73728, 3072, 3072) and bool(getenv("GPTOSS_DOWN_WGRAD_TAIL", 1)) and
      bool(getenv("GPTOSS_DOWN_WGRAD_EXPERT_COUNTS", 0))))
  if use_expert_counts:
    assert expert_counts is not None and expert_counts.dtype == dtypes.int32 and expert_counts.shape[-1] == n_experts, \
      f"expert counts must end in ({n_experts},) int32, got {expert_counts.shape} {expert_counts.dtype}"
  direct_shards = (is_multi and len(g.device) == 8 and ZERO2 and use_expert_counts and not do_bias and
                   not getenv("FUSED_WGRAD_QUANT", 0) and bool(getenv("GPTOSS_WGRAD_SHARDS", 1)))
  # A fused producer can guarantee a dense row-major output with the exact (M,N) shape this custom kernel reads.
  # Preserve that buffer identity instead of forcing a full bf16 copy before transpose-quantize.
  gc = g if g_direct else g.contiguous()
  if do_bias:
    binv = Tensor.invalids(1, n_experts * N, dtype=dtypes.float32, device=g.device)
    bout = Tensor(binv.uop.unshard(0), device=g.device) if is_multi else binv
  if getenv("FUSED_WGRAD_QUANT", 0):
    assert not do_bias, "bias fusion is implemented for prequantized wgrad"
    out = Tensor.custom_kernel(out, gc, xg.contiguous(), expert_off,
                               fxn=functools.partial(custom_hk_grouped_mxfp8_wgrad_fused, dname=dname, n_experts=n_experts))[0]
  else:
    if g_colw is not None: gT, g_si = g_colw
    elif do_tq_bias:
      from extra.llama_kernels.transpose_quantize_mxfp8 import transpose_quantize_mxfp8_bias
      gT, _, g_si, g_bias_partial = transpose_quantize_mxfp8_bias(gc)
    else: gT, _, g_si = transpose_quantize_mxfp8(gc)
    if xg_colw is not None: xT, x_si = xg_colw                       # pre-emitted in K1's epilogue (no transpose_quantize(xg))
    elif xg_rowwise is not None:
      from extra.llama_kernels.transpose_quantize_mxfp8 import transpose_requantize_mxfp8
      # Requantize rowwise MXFP8 directly to the exact (K,M) columnwise allocation consumed by wgrad. This avoids
      # both the dequantized BF16 HBM intermediate and a shape-fixing contiguous/copy between the custom kernels.
      xT, _, x_si = transpose_requantize_mxfp8(xg_rowwise[0], xg_rowwise[1])
    else: xT, _, x_si = transpose_quantize_mxfp8(xg.contiguous())
    if direct_shards:
      chunks = [Tensor(Tensor.invalids(1, 4*N, K, dtype=dtypes.bfloat16, device=g.device).uop.unshard(0), device=g.device)
                for _ in range(8)]
      wys = Tensor.custom_kernel(chunks[0], gT, xT, g_si, x_si, expert_off, expert_counts, *chunks[1:],
        fxn=functools.partial(custom_hk_grouped_mxfp8_wgrad, dname=dname, n_experts=n_experts, direct_shards=True))
      out = _reduce_scatter_expert_chunks([wys[0], *wys[7:14]])
    else:
      wargs = ((out, gT, xT, g_si, x_si, expert_off, bout, g) if do_bias else
               (out, gT, xT, g_si, x_si, expert_off)) + ((expert_counts,) if use_expert_counts else ())
      wys = Tensor.custom_kernel(*wargs, fxn=functools.partial(custom_hk_grouped_mxfp8_wgrad, dname=dname, n_experts=n_experts))
      out = wys[0]
      if do_bias: bout = wys[6]
  if direct_shards: pass  # Already reduced into the same expert-axis partition used by the optimizer.
  elif is_multi and ZERO2:
    # reduce-scatter over devices, sharded on the expert-row axis; after the model's per-layer [i] index this
    # assembles into a 4D grad sharded on the expert axis (axis 1), matching _zero_shard's ZeRO-2 partition.
    out = reduce_scatter_devaxis(out, 0)
  else:
    out = out.sum(0) if is_multi else out.squeeze(0)
  wret = out.reshape(n_experts, N, K)
  if do_tq_bias: return wret, _fc1_bias_grad_partials_hip(g_bias_partial, expert_off, n_experts)
  if not do_bias: return wret
  bret = (bout.sum(0) if is_multi else bout.squeeze(0)).reshape(n_experts, N)
  return wret, bret

def _fc1_handoff(aq:Tensor) -> tuple[UOp, int, bool, bool]|None:
  # Identify our producer by its callback and exact output slot, not a pre-substitution buffer identity.
  # @function renumbers fresh flat buffers; the current CALL still carries every saved output and count input.
  u = aq.uop
  for _ in range(64):
    if u.op is Ops.AFTER and u.src[1].op is Ops.CALL:
      call, grad_fxn = u.src[1], u.src[1].arg.grad_fxn
      if isinstance(grad_fxn, functools.partial) and grad_fxn.func is _fc1_swiglu_bwd and u.src[0] is call.src[2]:
        return (call, *grad_fxn.keywords['handoff'])
    if not u.src: break
    u = u.src[0]
  return None

def _fc1_dswiglu_operand(aq:Tensor) -> tuple[Tensor, int, Tensor|None]|None:
  if not FUSED_DGRAD_DSWIGLU or (ctx := _fc1_handoff(aq)) is None: return None
  call, real_inter, has_counts, _ = ctx
  counts = Tensor(call.src[-1], device=aq.device) if has_counts else None
  return Tensor(call.src[1].after(call), device=aq.device), real_inter, counts

def _fc1_colw_wgrad_operand(aq:Tensor) -> tuple[tuple[Tensor, Tensor], Tensor|None]|None:
  # FUSED_FC1_COLW down/FC2 wgrad hook. aq is the down-gemm's A input = pad(shrink(y_fp8)) where y_fp8 is the fused-FC1
  # swiglu kernel's row-major fp8 output. If that kernel also emitted the columnwise mxfp8, return the
  # PRE-EMITTED (xT (K, M) fp8, x_si packed e8) padded to the down-gemm K, so grouped_mx_wgrad(xg=None, xg_colw=...)
  # skips transpose_quantize(dequant(y_fp8)) -- byte-exact with it (proven in test_fc1_colw.py). y_colw is reconstructed
  # via out_yc.after(the swiglu CALL)[:real_inter], the exact UOp the model SAVED, so @function substitutes it and the
  # fused fwd is NOT re-run in the backward. Returns None (fall back to the x_phys path) if not a colw producer.
  if not FUSED_FC1_COLW or (ctx := _fc1_handoff(aq)) is None: return None
  call, real_inter, has_counts, has_colw = ctx
  if not has_colw: return None
  out_inter = call.src[2].shape[1]
  dev = aq.device
  yc_full = Tensor(call.src[12].after(call), device=dev)
  yce_full = Tensor(call.src[13].after(call), device=dev)
  counts = Tensor(call.src[-1], device=dev) if has_counts else None
  K = aq.shape[1]                                                      # the down-gemm K (== wgrad K, real_inter padded)
  if yce_full.dtype == dtypes.uint32:
    assert out_inter == K == 3072 and yc_full.shape == (K, aq.shape[0]) and yce_full.shape == (aq.shape[0] // 128, K)
    return (yc_full, yce_full), counts
  if out_inter == K: return (yc_full, mx_pack(yce_full)), counts
  yc, yce = yc_full[:real_inter], yce_full[:real_inter]
  if K != real_inter:                                                 # columnwise-of-zero pad rows == transpose_quantize's zero-pad rows
    yc  = yc.pad(((0, K - real_inter), (0, 0)))
    yce = yce.pad(((0, K - real_inter), (0, 0)))
  return (yc.contiguous(), mx_pack(yce.contiguous())), counts

def mx_pack_3d(e8:Tensor) -> Tensor:
  E, rows, scale_K = e8.shape
  return e8.reshape(E, rows, scale_K // 4, 4).bitcast(dtypes.uint32).reshape(E, rows, scale_K // 4).permute(0, 2, 1).contiguous()

# ---- FUSED_DOWN_EPILOGUE: fold the down-gemm's per-expert bias + router-weighted COMBINE scatter into the epilogue --
# (gated, default off). The down gemm writes z (byte-exact, saved for the backward's d_weights) AND atomic-scatters the
# router-weighted contribution straight into the token output row -> eliminates the separate `+dn_bias` elementwise and
# the `combine` gather (its z re-read). The scatter accumulates in fp32 (order-dependent across the k experts per token,
# so `out` is ~bf16-eps close, NOT byte-exact with the deterministic (sel*weights).sum(k) combine). z CANNOT be dropped:
# the router-weight gradient d_weights = <d_out, z> needs it.

def _fill_neg1_kernel(out:UOp) -> UOp:
  i = UOp.range(out.numel(), 0)
  return out.flatten().index(i).store(UOp.const(-1, out.dtype)).end(i).sink(arg=KernelInfo(name="fill_neg1"))

def _row_meta_kernel(row_tok:UOp, row_w:UOp, dest_row:UOp, weights:UOp) -> UOp:
  # Inverse of dest_row (a dropless bijection): for each dispatch slot s -> grouped row=dest_row[g,s], token t=s//k,
  # slot j=s%k. Writes row_tok[g,row]=t and row_w[g,row]=weights[g,t,j]. Unwritten (pad) rows keep the -1 / 0 init,
  # so the epilogue's `dest_token >= 0` guard skips them. Contention-free (each row is the target of one slot).
  G, Sk = dest_row.shape
  k = weights.shape[2]
  g = UOp.range(G, 0)
  s = UOp.range(Sk, 1)
  row = dest_row.index(g, s).cast(dtypes.weakint)
  t = (s // k).cast(dtypes.weakint)
  j = (s % k).cast(dtypes.weakint)
  wv = weights.index(g, t, j).load()
  s1 = row_tok.index(g, row).store((s // k).cast(row_tok.dtype))
  s2 = row_w.after(s1).index(g, row).store(wv.cast(row_w.dtype))
  return s2.end(s, g).sink(arg=KernelInfo(name=f"row_meta_{Sk}", opts_to_apply=()))

def _build_row_meta(r) -> tuple[Tensor, Tensor]:
  from extra.gemm.moe_routing import _sharded_invalids, _sharded_zeros
  dev = r.dest_row.device
  G, m_l = r.n_groups, r.m_l
  row_tok = Tensor.custom_kernel(_sharded_invalids((G, m_l), dtypes.int32, dev), fxn=_fill_neg1_kernel)[0]
  row_w = _sharded_zeros((G, m_l), dtypes.bfloat16, dev)
  row_tok, row_w, *_ = Tensor.custom_kernel(row_tok, row_w, r.dest_row, r.weights, fxn=_row_meta_kernel)
  return row_tok, row_w

@functools.cache
def custom_hk_grouped_mxfp8_gemm_down_combine(Out:UOp, Z:UOp, DownBias:UOp, DestTok:UOp, RowW:UOp, DestRow:UOp, Weights:UOp,
                                              A:UOp, B:UOp, scale_A:UOp, scale_B:UOp, *extra:UOp,
                                              dname:str, n_experts:int, nreal:int) -> UOp:
  M, K = A.shape
  E, N, K2 = B.shape
  assert K == K2, f"{A.shape} {B.shape}"
  assert E == n_experts, f"{E} != {n_experts}"
  threads = UOp.special(64 * 8, "lidx0")
  workgroups = UOp.special((M // 256) * (N // 256), "gidx0")
  sink_inputs = (Out.base, Z.base, DownBias.base, DestTok.base, RowW.base, DestRow.base, Weights.base,
                 A.base, B.base, scale_A.base, scale_B.base, extra[0].base, extra[1].base, extra[2].base, threads, workgroups)
  sink = UOp.sink(*sink_inputs,
                  arg=KernelInfo(f"hk_grouped_mxfp8_gemm_down_combine_{E}_{M}_{N}_{K}",
                                 estimates=Estimates(ops=2*M*N*K, mem=(M*K+E*N*K)*A.dtype.itemsize+M*nreal*Z.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens_path/"grouped_mxfp8_gemm_down_combine.cpp").read_text()
  # NOTE: DOUBLE_BUFFER corrupts this kernel's register epilogue (non-deterministic accumulator garbage; the base
  # gemm's kittens store and the swiglu epilogue survive DBUF but this direct-read epilogue does not, and no
  # post-loop drain/barrier fixes it). Single-buffer (default) is byte-exact. Gated on DOWN_DBUF for future repair.
  dbuf = ["-DDOUBLE_BUFFER=1"] if getenv("DOWN_DBUF", 0) else []
  dbg = ([f"-DNO_SCATTER={getenv('NO_SCATTER',0)}"] if getenv("NO_SCATTER",0) else []) + \
        ([f"-DNO_ZWRITE={getenv('NO_ZWRITE',0)}"] if getenv("NO_ZWRITE",0) else [])
  lib = HIPCCCompiler("gfx950", [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}",
                                 f"-DGEMM_E={E}", f"-DGEMM_NREAL={nreal}", *dbuf, *dbg]).compile_cached(src)
  return hip_program(sink, src, lib)

def _down_combine_bwd(gradient:UOp, kernel:UOp) -> tuple:
  # backward for the fused down+bias+combine: reconstruct (combine bwd -> down-gemm dgrad/wgrad + down bias grad).
  # srcs: (out, z, down_bias, dest_token, row_weight, dest_row, weights, x_q, w_q, x_si, w_si, xe_in, w_e8, expert_off)
  from extra.gemm.moe_routing import grouped_gather_rows, _sharded_zeros, _sharded_invalids
  from extra.llama_kernels.fused_combine_gptoss import (_custom_combine_bwd_dz, _custom_combine_bwd_dz_gather,
    _custom_combine_bwd_dz_hip, _custom_combine_bwd_dz_dual, _build_src_row, COMBINE_GATHER, COMBINE_DZ_HIP,
    COMBINE_DZ_WRITEONLY, _combine_dz_output, BLK)
  inputs = kernel.src[1:]
  dev = inputs[1].device
  z3 = Tensor(inputs[1].after(kernel), device=dev)                       # SAVED z (G, m_l, D)
  G, m_l, D = z3.shape
  destT = Tensor(inputs[5], device=dev)                                  # dest_row (G, T_l*k)
  wT = Tensor(inputs[6], device=dev)                                     # weights (G, T_l, k)
  T_l, k = wT.shape[1], wT.shape[2]
  aq = Tensor(inputs[7], device=dev)
  bq = Tensor(inputs[8], device=dev)
  ae8 = Tensor(inputs[11], device=dev)
  be8 = Tensor(inputs[12], device=dev)
  expert_off = Tensor(inputs[13], device=dev)
  E, N = bq.shape[0], bq.shape[1]
  M, Kk = aq.shape
  d_out = Tensor(gradient, device=dev).reshape(G, T_l, D).cast(dtypes.bfloat16).contiguous()
  # --- combine backward: d_weights = <d_out, gathered z>, d_z = scatter(d_out * weight) quantized (like _combine_bwd)
  sel = grouped_gather_rows(z3, destT, G).reshape(G, T_l, k, D)
  dw = (sel.cast(dtypes.float) * d_out.reshape(G, T_l, 1, D)).sum(3)
  dual_dz = bool(getenv("FUSED_DZ_DUAL", 0))
  dz_phys = N if getenv("FUSED_DZ_COLW_SEPARATE", 0) or dual_dz else D
  if dual_dz:
    dz_bf16 = _sharded_invalids((G, m_l, dz_phys), dtypes.bfloat16, dev)
    dz_fp8 = _sharded_invalids((G, m_l, dz_phys), FP8_DTYPE, dev)
    dz_e8 = _sharded_invalids((G, m_l, dz_phys // BLK), dtypes.uint8, dev)
    dz_col = alloc_like((dz_phys, G * m_l), FP8_DTYPE, dev, 1)
    dz_col_si = alloc_like((G * m_l // 128, dz_phys), dtypes.uint32, dev, 0)
  else:
    dz_alloc = _combine_dz_output if COMBINE_GATHER and COMBINE_DZ_WRITEONLY else _sharded_zeros
    dz_bf16 = dz_alloc((G, m_l, dz_phys), dtypes.bfloat16, dev)
    dz_fp8 = dz_alloc((G, m_l, dz_phys), FP8_DTYPE, dev)
    dz_e8 = dz_alloc((G, m_l, dz_phys // BLK), dtypes.uint8, dev)
  if COMBINE_GATHER:  # coalesced-write gather via the inverse map (~2.5x the scatter); byte-exact
    src_row = _build_src_row(destT, m_l)
    is_amd = (dev[0] if isinstance(dev, tuple) else dev).split(":")[0] == "AMD"
    if dual_dz:
      dz_bf16, dz_fp8, dz_e8, dz_col, dz_col_si, *_ = Tensor.custom_kernel(
        dz_bf16, dz_fp8, dz_e8, dz_col, dz_col_si, d_out, src_row, wT, fxn=_custom_combine_bwd_dz_dual)
    else:
      dz_fxn = _custom_combine_bwd_dz_hip if COMBINE_DZ_HIP and is_amd else _custom_combine_bwd_dz_gather
      dz_bf16, dz_fp8, dz_e8, *_ = Tensor.custom_kernel(dz_bf16, dz_fp8, dz_e8, d_out, src_row, wT, fxn=dz_fxn)
  else:
    assert not dual_dz, "FUSED_DZ_DUAL requires COMBINE_GATHER"
    dz_bf16, dz_fp8, dz_e8, *_ = Tensor.custom_kernel(dz_bf16, dz_fp8, dz_e8, d_out, destT, wT, fxn=_custom_combine_bwd_dz)
  # --- down-gemm backward (w_stored path): dgrad via the (fp8,e8) tuple, wgrad + bias over d_z (padded D -> N)
  g_fp8, g_e8 = dz_fp8.reshape(M, dz_phys), dz_e8.reshape(M, dz_phys // BLK)
  g_bf16 = dz_bf16.reshape(M, dz_phys)
  # Reuse the packed down-weight scales already consumed by the forward GEMM. Extending that small buffer's
  # lifetime avoids the requantizer's scattered raw-e8 loads without adding another packing kernel.
  w_op = _dgrad_wT_operand(inputs[8]) or _dynamic_dgrad_wT_operand(bq, be8, Tensor(inputs[10], device=aq.device))
  if (fc1_ctx := _fc1_dswiglu_operand(aq)) is not None:
    grad_x = grouped_mx_gemm_dgrad_dswiglu((g_fp8, g_e8), w_op, expert_off, *fc1_ctx)
  else:
    grad_x = grouped_mx_gemm((g_fp8, g_e8), w_op, expert_off, logical_k=D)
  gw_g = g_bf16.pad(((0, 0), (0, N - dz_phys))) if N != dz_phys else g_bf16
  g_colw = (dz_col, dz_col_si) if dual_dz else None
  if getenv("FUSED_DZ_COLW_SEPARATE", 0) and not dual_dz:
    from extra.llama_kernels.transpose_quantize_mxfp8 import transpose_quantize_mxfp8
    dz_col, _, dz_col_si = transpose_quantize_mxfp8(gw_g)
    g_colw = (dz_col, dz_col_si)
  # FUSED_FC1_COLW: read the pre-emitted columnwise activation, skipping transpose_quantize+dequant.
  if (colw_ctx := _fc1_colw_wgrad_operand(aq)) is not None:
    colw, expert_counts = colw_ctx
    grad_w = grouped_mx_wgrad(gw_g, None, expert_off, E, xg_colw=colw, g_colw=g_colw, expert_counts=expert_counts)
  else:
    x_phys = (aq.cast(dtypes.bfloat16) * _mx_block_scale(ae8).cast(dtypes.bfloat16))
    grad_w = grouped_mx_wgrad(gw_g, x_phys, expert_off, E, g_colw=g_colw)
  grad_xq = grad_x if fc1_ctx is not None else grad_x * _mx_block_scale(ae8).cast(dtypes.bfloat16)
  grad_wq = grad_w.contiguous()
  d_bias = _fc1_bias_grad(g_bf16[:, :D], expert_off, E)                   # (E, D) per-expert sum of real d_z
  # grads for (out, z, down_bias, dest_token, row_weight, dest_row, weights, x_q, w_q, x_si, w_si, xe_in, w_e8, expert_off)
  return (None, None, d_bias.uop, None, None, None, dw.cast(wT.dtype).uop, grad_xq.uop, grad_wq.uop, None, None, None, None, None)

def grouped_mx_gemm_down_combine(x:Tensor|tuple[Tensor, Tensor], w:Tensor|tuple[Tensor, Tensor], down_bias:Tensor, r,
                                 n_tokens:int, experts_per_tok:int, real_out:int) -> tuple[Tensor, Tensor]:
  # Fused DOWN gemm + per-expert bias + router-weighted combine scatter. Input prep byte-identical to grouped_mx_gemm.
  # down_bias is (E, real_out) bf16 (per-expert). Returns (out (n_tokens, real_out) bf16, z (M, real_out) bf16 [saved]).
  from extra.gemm.moe_routing import _sharded_zeros, _sharded_invalids
  if (pre_quantized := isinstance(w, tuple)):
    if len(w) == 4:                                        # PRESTORE_WT: 4-tuple (w_q,w_e8,wT_q,wT_e8); register W^T for the dgrad
      w_q, w_e8, _wT_q, _wT_e8 = w
      _dgrad_wT_mailbox[w_q.uop] = (_wT_q, _wT_e8)
    else:
      w_q, w_e8 = w
    E, N, K2 = w_q.shape
  else:
    E, N, K2 = w.shape
  if isinstance(x, tuple):
    xq_raw, xe_raw = x
    xref, M, Kin = xq_raw, *xq_raw.shape
    x_q = xq_raw.pad(((0, 0), (0, K2 - Kin))) if K2 != Kin else xq_raw
    x_e8 = xe_raw.pad(((0, 0), (0, K2 // 32 - Kin // 32)), value=127) if K2 != Kin else xe_raw
    K = K2
  else:
    xref, (M, K) = x, x.shape
    x_q, x_e8, _ = quantize_mxfp8(x)
  x_si = mx_pack(x_e8)
  assert K == K2, f"shape mismatch K {K} != {K2}"
  assert M % 256 == 0 and N % 256 == 0 and K % 128 == 0, f"grouped mxfp8 down needs M%256,N%256,K%128, got {M,K}"
  assert real_out <= N and real_out % 32 == 0, f"real_out {real_out} must be <= {N} and mult of 32"
  assert down_bias.shape == (E, real_out) and down_bias.dtype == dtypes.bfloat16, f"down_bias must be (E,{real_out}) bf16, got {down_bias.shape}"
  down_bias = down_bias.contiguous()   # w_down_bias[i] is a strided view; the epilogue indexes it row-major (E, real_out)
  dname = (xref.device[0] if isinstance(xref.device, tuple) else xref.device).split(":")[0]
  if not pre_quantized: w_q, w_e8, _ = quantize_mxfp8(w)
  w_si = mx_pack_3d(w_e8)
  xe_in = x_e8.reshape(M, K // 32)
  G, dev = r.n_groups, xref.device
  row_tok, row_w = _build_row_meta(r)
  # out: fp32 token accumulator (zero-init for the atomic scatter), sharded on the group axis like combine's out.
  # z: (G, m_l, real_out) bf16 grouped rows, sharded like the gemm rows (device g holds group g's m_l rows == out's group).
  out = _sharded_zeros((G, r.t_local, real_out), dtypes.float32, dev)
  z = _sharded_invalids((G, r.m_l, real_out), dtypes.bfloat16, dev)
  ys = Tensor.custom_kernel(out, z, down_bias, row_tok, row_w, r.dest_row, r.weights,
                            x_q, w_q, x_si, w_si, xe_in, w_e8, r.off,
                            fxn=functools.partial(custom_hk_grouped_mxfp8_gemm_down_combine, dname=dname, n_experts=E, nreal=real_out),
                            grad_fxn=_down_combine_bwd)
  # ys[1] (the RAW z output, G x m_l x real_out) is returned for the caller to SAVE: the backward reads it via
  # inputs[1].after(kernel), so saving the same UOp lets @function substitute it (no fused-fwd re-run in the backward).
  return ys[0].cast(dtypes.bfloat16).reshape(n_tokens, real_out), ys[1]

def _dgrad_fp8_lookup(gradient:UOp):
  # recover the swiglu-bwd (fp8,e8) companion for this dgrad's incoming grad: strip the pad/reshape the
  # [:, :2*inter] shrink-backward inserts between the swiglu bwd's gh and here. Returns None on miss (falls
  # back to the bf16 quantize path). Only the gate_up dgrad's grad (=d_h) is ever in the mailbox.
  from extra.llama_kernels.fused_swiglu_quantize_gptoss import _moe_dgrad_mailbox, FUSED_DGRAD_FP8
  if not FUSED_DGRAD_FP8 or not _moe_dgrad_mailbox: return None
  seen, stack, depth = set(), [gradient], 0
  while stack and depth < 64:
    n = stack.pop()
    depth += 1
    if id(n) in seen: continue
    seen.add(id(n))
    if (hit := _moe_dgrad_mailbox.get(n)) is not None: return hit
    b = n.base if n is not n.base else None
    if b is not None and (hit := _moe_dgrad_mailbox.get(b)) is not None: return hit
    if n.op in (Ops.PAD, Ops.RESHAPE, Ops.SHRINK, Ops.STAGE, Ops.EXPAND, Ops.CAST) and n.src: stack.append(n.src[0])
    if b is not None: stack.append(b)
  return None

@functools.cache
def custom_grouped_mx_gemm_bw(gradient:UOp, kernel:UOp, w_stored:bool=False, return_bias:bool=False) -> tuple:
  inputs = kernel.src[1:]
  aq = Tensor(inputs[1], device=inputs[1].device)
  bq = Tensor(inputs[2], device=inputs[2].device)
  ae8 = Tensor(inputs[5], device=inputs[5].device)
  be8 = Tensor(inputs[6], device=inputs[6].device)
  E, N = bq.shape[0], bq.shape[1]
  M, K = aq.shape
  g = Tensor(gradient, device=aq.device).reshape(M, N).cast(dtypes.bfloat16)
  w_op = _dgrad_wT_operand(inputs[2]) or _dynamic_dgrad_wT_operand(bq, be8, Tensor(inputs[4], device=aq.device))
  expert_off = Tensor(inputs[7], device=inputs[7].device)
  if (g_pre := _dgrad_fp8_lookup(gradient)) is not None:
    # swiglu bwd already emitted d_h as mxfp8 (unpadded on N); feed the tuple path so the dgrad skips re-quantizing g
    g_in = (Tensor(g_pre[0], device=aq.device), Tensor(g_pre[1], device=aq.device))
  else:
    g_in = g
  g_colw, g_bias = None, None
  if g_pre is not None and len(g_pre) == 5:
    g_scale = Tensor(g_pre[3], device=aq.device)
    assert g_scale.dtype in (dtypes.uint8, dtypes.uint32)
    g_colw = (Tensor(g_pre[2], device=aq.device), g_scale if g_scale.dtype == dtypes.uint32 else mx_pack(g_scale))
    g_bias = Tensor(g_pre[4], device=aq.device)
    assert return_bias, "no-BF16 combine requires the down-bias backward consumer"
  if getenv("GPTOSS_COMBINE_DZ_QUANT", 0) and return_bias:
    assert g_colw is not None and g_bias is not None, "lost no-BF16 combine companions"
  if (fc1_ctx := _fc1_dswiglu_operand(aq)) is not None:
    grad_x = grouped_mx_gemm_dgrad_dswiglu(g_in, w_op, expert_off, *fc1_ctx)
  else:
    grad_x = grouped_mx_gemm(g_in, w_op, expert_off,
                             logical_k=2880 if (E, M, N, K) == (32, 73728, 3072, 3072) else None)
  # FUSED_FC1_COLW: read the pre-emitted columnwise, skipping transpose_quantize+dequant.
  if (colw_ctx := _fc1_colw_wgrad_operand(aq)) is not None:
    colw, expert_counts = colw_ctx
    grad_w = grouped_mx_wgrad(g, None, expert_off, E, xg_colw=colw, g_colw=g_colw, bias_grad=return_bias and g_bias is None,
                              g_direct=g_pre is not None, expert_counts=expert_counts)
  else:
    if getenv("FUSED_XG_REQUANT", 0):
      grad_w = grouped_mx_wgrad(g, None, expert_off, E, xg_rowwise=(aq, ae8), g_colw=g_colw,
                                bias_grad=return_bias and g_bias is None, g_direct=g_pre is not None)
    else:
      x_phys = (aq.cast(dtypes.bfloat16) * _mx_block_scale(ae8).cast(dtypes.bfloat16))
      grad_w = grouped_mx_wgrad(g, x_phys, expert_off, E, g_colw=g_colw, bias_grad=return_bias and g_bias is None, g_direct=g_pre is not None)
  grad_xq = grad_x if fc1_ctx is not None else grad_x * _mx_block_scale(ae8).cast(dtypes.bfloat16)
  grad_w, d_bias = grad_w if isinstance(grad_w, tuple) else (grad_w, None)
  if g_bias is not None: d_bias = _fc1_bias_grad_partials_hip(g_bias, expert_off, E)
  grad_wq = _finish_expert_wgrad(grad_w) if w_stored else (grad_w * _mx_block_scale_3d(be8).cast(dtypes.bfloat16)).contiguous()
  ret = (None, grad_xq.uop, grad_wq.uop) + tuple(None for _ in inputs[3:])
  return (ret, d_bias) if return_bias else ret

_grouped_bw_stored = functools.partial(custom_grouped_mx_gemm_bw, w_stored=True)

def grouped_mx_gemm(x:Tensor|tuple[Tensor, Tensor], w:Tensor|tuple[Tensor, ...], expert_off:Tensor,
                    out_scale:Tensor|None=None, logical_k:int|None=None) -> Tensor:
  prepacked_w_si, packed_x_si = None, False
  if (pre_quantized := isinstance(w, tuple)):
    if len(w) == 4:                                        # PRESTORE_WT: 4-tuple (w_q,w_e8,wT_q,wT_e8); register W^T for the dgrad
      w_q, w_e8, _wT_q, _wT_e8 = w
      _dgrad_wT_mailbox[w_q.uop] = (_wT_q, _wT_e8)
    elif len(w) == 3:
      w_q, w_e8, prepacked_w_si = w
    else:
      w_q, w_e8 = w
    E, N, K2 = w_q.shape
  else:
    E, N, K2 = w.shape
  if isinstance(x, tuple):
    # PRE-QUANTIZED activation (fp8, e8) from a fused producer (e.g. fused_swiglu_quantize), UNPADDED on K.
    # pad to the weight's K (mult of 128): fp8 pad = 0, e8 pad = 127 (dequant scale 1, 0*1=0), then pack si.
    # grad flows to xq_raw (the fused fp8 output) via the pad's shrink-backward; the gemm's grad_xq*block_scale
    # and the producer's grad*qscale are reciprocals that cancel -> physical grad reaches the producer.
    xq_raw, xe_raw = x
    xref, M, Kin = xq_raw, *xq_raw.shape
    x_q = xq_raw.pad(((0, 0), (0, K2 - Kin))) if K2 != Kin else xq_raw
    packed_x_si = xe_raw.dtype == dtypes.uint32
    if packed_x_si:
      assert (E, N, K2, Kin) == (32, 3072, 5888, 5888) and M % 73728 == 0
      assert xe_raw.shape == (46, M) and out_scale is not None
      x_si = xe_raw
    else:
      x_e8 = xe_raw.pad(((0, 0), (0, K2 // 32 - Kin // 32)), value=127) if K2 != Kin else xe_raw
      x_si = mx_pack(x_e8)
    K = K2
  else:
    xref, (M, K) = x, x.shape
    x_q, x_e8, x_si = quantize_mxfp8(x)
  assert K == K2, f"shape mismatch K {K} != {K2}"
  assert logical_k is None or 0 < logical_k <= K, f"logical K {logical_k} must be in (0,{K}]"
  assert M % 256 == 0 and N % 256 == 0 and K % 128 == 0, f"grouped mxfp8 needs M%256,N%256,K%128, got {M,K} {w_q.shape if pre_quantized else w.shape}"
  dname = (xref.device[0] if isinstance(xref.device, tuple) else xref.device).split(":")[0]
  if not pre_quantized: w_q, w_e8, _ = quantize_mxfp8(w)
  w_si = prepacked_w_si if prepacked_w_si is not None else mx_pack_3d(w_e8)
  # The exact FC1 dgrad never reads the raw e8 ABI slot; alias its already-packed scale input.
  xe_in, out_shape = x_si if packed_x_si else x_e8.reshape(M, K // 32), (M, N)
  out = alloc_like(out_shape, dtypes.bfloat16, xref.device, xref.uop.axis)
  if out_scale is not None:
    if out_scale.dtype == dtypes.uint32:
      assert (E, N, K) == (32, 3072, 5888) and M % 73728 == 0 and out_scale.shape == (N // 128, M), \
        f"packed GPT-OSS out_scale must be uint32 ({N // 128},{M}), got {out_scale.dtype} {out_scale.shape}"
    else:
      assert out_scale.shape == (M, N // 32) and out_scale.dtype == dtypes.uint8, \
        f"out_scale must be uint8 ({M},{N // 32}), got {out_scale.dtype} {out_scale.shape}"
  return Tensor.custom_kernel(out, x_q, w_q, x_si, w_si, xe_in, w_e8, expert_off, *((out_scale,) if out_scale is not None else ()),
                              fxn=functools.partial(custom_hk_grouped_mxfp8_gemm, dname=dname, n_experts=E, logical_k=logical_k),
                              grad_fxn=(_grouped_bw_stored if pre_quantized else custom_grouped_mx_gemm_bw))[0]

# ---- FUSED_DOWN_BIAS: fold ONLY the per-expert down-gemm bias into the base grouped-gemm epilogue -----------------
# (gated, default off). The one cheap thing K2's down epilogue does: a per-column register add (no quantize, no
# transpose, no scatter). Adds down_bias[e, col] to the fp32 accumulator BEFORE the single bf16 store round, so the
# fused output is MORE accurate than the separate `+dn_bias` path (which rounds z to bf16, then adds bias, then rounds
# again). Eliminates the separate bias elementwise kernel and its full z round-trip. The COMBINE stays separate.

@functools.cache
def custom_hk_grouped_mxfp8_gemm_down_bias(C:UOp, A:UOp, B:UOp, scale_A:UOp, scale_B:UOp, *extra:UOp,
                                           dname:str, n_experts:int, nreal:int) -> UOp:
  # same as custom_hk_grouped_mxfp8_gemm but compiled with -DFUSED_DOWN_BIAS and one extra kernel input (down_bias,
  # E x nreal bf16) appended after expert_off. extra = (xe_in, w_e8, expert_off, down_bias).
  M, K = A.shape
  E, N, K2 = B.shape
  assert K == K2, f"{A.shape} {B.shape}"
  assert E == n_experts, f"{E} != {n_experts}"
  gptoss_logical_k = (E, M, N, K, nreal) == (32, 73728, 3072, 3072, 2880) and bool(getenv("GPTOSS_DOWN_LOGICAL_K", 1))
  gptoss_out_tail = gptoss_logical_k and bool(getenv("DBUF", 1)) and bool(getenv("GPTOSS_DOWN_FWD_OUT_TAIL", 1))
  # The exact fused down-forward keeps its ping-pong LDS bases scalar and uses five uniform expert probes. On two
  # MI350X GPUs the pair was full-buffer byte-exact and 5.57-6.15% faster (249 VGPR, 135168B LDS, no spills).
  gptoss_scalar_lds = gptoss_out_tail and bool(getenv("GPTOSS_DOWN_FWD_SCALAR_LDS", 1))
  gptoss_binary_expert = gptoss_out_tail and bool(getenv("GPTOSS_DOWN_FWD_BINARY_EXPERT", 1))
  # RT_B has only two live scale bytes; the exact down-forward needs no upper two scale-row loads.
  gptoss_pack_b32 = gptoss_out_tail and bool(getenv("GPTOSS_DOWN_FWD_PACK_B32", 1))
  true_grid_requested = getenv("GPTOSS_DOWN_FWD_TRUE_GRID", 0)
  assert true_grid_requested in (0, 1), f"GPTOSS_DOWN_FWD_TRUE_GRID must be 0 or 1, got {true_grid_requested}"
  gptoss_true_grid = bool(true_grid_requested) and gptoss_pack_b32 and gptoss_scalar_lds and gptoss_binary_expert \
    and getenv("GROUPED_WGM", 8) == 8 and getenv("MOE_SKIP_EMPTY", 0) == 1 and getenv("MOE_SKIP_NO_ZERO", 0) == 0
  executed_k = 2944 if gptoss_logical_k else K
  executed_n = 2880 if gptoss_out_tail else N
  threads = UOp.special(64 * 8, "lidx0")
  workgroups = UOp.special((M // 256) * (N // 256), "gidx0")
  sink_inputs = (C.base, A.base, B.base, scale_A.base, scale_B.base,
                 extra[0].base, extra[1].base, extra[2].base, extra[3].base, threads, workgroups)
  sink = UOp.sink(*sink_inputs,
                  arg=KernelInfo(f"hk_grouped_mxfp8_gemm_down_bias_logicalk{int(gptoss_logical_k)}"+
                                 (f"_outtail{int(gptoss_out_tail)}" if gptoss_logical_k else "")+
                                 (f"_lds{int(gptoss_scalar_lds)}_bin{int(gptoss_binary_expert)}" if gptoss_out_tail else "")+
                                 ("_b32" if gptoss_pack_b32 else "")+
                                 ("_tg" if gptoss_true_grid else "")+
                                 f"_{E}_{M}_{N}_{K}",
                                 estimates=Estimates(ops=2*M*executed_n*executed_k,
                                                     mem=(M*K+E*N*K)*A.dtype.itemsize+M*N*C.dtype.itemsize)))
  kittens_path = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (kittens_path/"grouped_mxfp8_gemm.cpp").read_text()
  dbuf = ["-DDOUBLE_BUFFER=1"] if getenv("DBUF", 1) else []
  lib = HIPCCCompiler("gfx950", [f"-I{(kittens_path/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DGEMM_M={M}", f"-DGEMM_N={N}", f"-DGEMM_K={K}",
                                 f"-DGEMM_E={E}", "-DFUSED_DOWN_BIAS=1", f"-DGEMM_NREAL={nreal}",
                                 f"-DGPTOSS_DOWN_LOGICAL_K={int(gptoss_logical_k)}", "-DGPTOSS_DOWN_REAL_K=2880",
                                 f"-DGPTOSS_DOWN_OUT_TAIL={int(gptoss_out_tail)}", "-DGPTOSS_DOWN_REAL_N=2880",
                                 "-DGPTOSS_DOWN_DGRAD_SCALAR_LDS=0",
                                 f"-DGPTOSS_DOWN_FWD_SCALAR_LDS={int(gptoss_scalar_lds)}",
                                 f"-DGPTOSS_DOWN_FWD_BINARY_EXPERT={int(gptoss_binary_expert)}",
                                 f"-DGPTOSS_DGRAD_PACK_B32={int(gptoss_pack_b32)}",
                                 f"-DGPTOSS_DOWN_FWD_TRUE_GRID={int(gptoss_true_grid)}",
                                 f"-DMOE_SKIP_EMPTY={getenv('MOE_SKIP_EMPTY', 0)}", f"-DMOE_SKIP_NO_ZERO={getenv('MOE_SKIP_NO_ZERO', 0)}",
                                 f"-DGROUPED_WGM={getenv('GROUPED_WGM', 8)}", *dbuf]).compile_cached(src)
  return hip_program(sink, src, lib)

def _down_bias_bwd(gradient:UOp, kernel:UOp) -> tuple:
  # backward for the fused down+bias gemm: the base gemm dgrad/wgrad (w_stored, byte-identical to the non-fused down
  # gemm) + the per-expert bias grad d_bias[e,col] = sum over expert-e rows of d_z[:, col] (allreduced under DP by
  # _fc1_bias_grad, exactly like the separate _moe_bias_tile bias grad). inputs:
  # (out, x_q, w_q, x_si, w_si, xe_in, w_e8, expert_off, down_bias)
  if getenv("FUSED_DZ_BIAS_PARTIAL", 0):
    base_raw, d_bias = custom_grouped_mx_gemm_bw(gradient, kernel, w_stored=True, return_bias=True)
    base = list(base_raw)
  else: base = list(custom_grouped_mx_gemm_bw(gradient, kernel, w_stored=True))
  inputs = kernel.src[1:]
  bq = Tensor(inputs[2], device=inputs[2].device)
  E, N = bq.shape[0], bq.shape[1]
  M = Tensor(inputs[1], device=inputs[1].device).shape[0]
  real_out = inputs[8].shape[1]                                                   # down_bias is (E, real_out)
  expert_off = Tensor(inputs[7], device=inputs[7].device)
  g = Tensor(gradient, device=inputs[1].device).reshape(M, N).cast(dtypes.bfloat16)
  if not getenv("FUSED_DZ_BIAS_PARTIAL", 0): d_bias = _fc1_bias_grad(g[:, :real_out], expert_off, E)
  base[-1] = d_bias[:, :real_out].uop                                               # replace the down_bias None slot
  return tuple(base)

def grouped_mx_gemm_down_bias(x:Tensor|tuple[Tensor, Tensor], w:Tensor|tuple[Tensor, Tensor], expert_off:Tensor,
                             down_bias:Tensor, real_out:int) -> Tensor:
  # grouped_mx_gemm with the per-expert down bias (E, real_out) bf16 folded into the epilogue. Input prep is
  # byte-identical to grouped_mx_gemm; returns the full (M, N) bf16 output (caller slices [:, :real_out]). Columns
  # >= real_out are unbiased padding (sliced off). The bias grad flows to down_bias via the grad_fxn.
  if (pre_quantized := isinstance(w, tuple)):
    if len(w) == 4:                                        # PRESTORE_WT: 4-tuple (w_q,w_e8,wT_q,wT_e8); register W^T for the dgrad
      w_q, w_e8, _wT_q, _wT_e8 = w
      _dgrad_wT_mailbox[w_q.uop] = (_wT_q, _wT_e8)
    else:
      w_q, w_e8 = w
    E, N, K2 = w_q.shape
  else:
    E, N, K2 = w.shape
  if isinstance(x, tuple):
    xq_raw, xe_raw = x
    xref, M, Kin = xq_raw, *xq_raw.shape
    x_q = xq_raw.pad(((0, 0), (0, K2 - Kin))) if K2 != Kin else xq_raw
    x_e8 = xe_raw.pad(((0, 0), (0, K2 // 32 - Kin // 32)), value=127) if K2 != Kin else xe_raw
    x_si = mx_pack(x_e8)
    K = K2
  else:
    xref, (M, K) = x, x.shape
    x_q, x_e8, x_si = quantize_mxfp8(x)
  assert K == K2, f"shape mismatch K {K} != {K2}"
  assert M % 256 == 0 and N % 256 == 0 and K % 128 == 0, f"grouped mxfp8 down needs M%256,N%256,K%128, got {M,K}"
  assert real_out <= N and real_out % 32 == 0, f"real_out {real_out} must be <= {N} and mult of 32"
  assert down_bias.shape == (E, real_out) and down_bias.dtype == dtypes.bfloat16, \
    f"down_bias must be (E,{real_out}) bf16, got {down_bias.shape} {down_bias.dtype}"
  down_bias = down_bias.contiguous()   # w_down_bias[i] is a strided view; the epilogue indexes it row-major (E, real_out)
  dname = (xref.device[0] if isinstance(xref.device, tuple) else xref.device).split(":")[0]
  if not pre_quantized: w_q, w_e8, _ = quantize_mxfp8(w)
  w_si = mx_pack_3d(w_e8)
  xe_in = x_e8.reshape(M, K // 32)
  out = alloc_like((M, N), dtypes.bfloat16, xref.device, xref.uop.axis)
  return Tensor.custom_kernel(out.contiguous(), x_q, w_q, x_si, w_si, xe_in, w_e8, expert_off, down_bias,
                              fxn=functools.partial(custom_hk_grouped_mxfp8_gemm_down_bias, dname=dname, n_experts=E, nreal=real_out),
                              grad_fxn=_down_bias_bwd)[0]
