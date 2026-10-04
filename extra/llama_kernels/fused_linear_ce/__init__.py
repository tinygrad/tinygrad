from __future__ import annotations
import functools, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.hipcc import HIPCCCompiler, rocm_toolchain, hip_program
from extra.llama_kernels import alloc_like, dname_of

PART_N, PART_M = 32, 8

@functools.cache
def _custom_lmhead_ce(partial_max:UOp, partial_sum:UOp, partial_target:UOp, saved_logits:UOp,
                      hidden:UOp, weight:UOp, weight_extra:UOp, targets:UOp, *, dname:str) -> UOp:
  M, K = hidden.shape
  V, K2 = weight.shape
  assert K == K2 and M % 256 == V % 256 == K % 64 == 0
  wgm = getenv("GPTOSS_LMHEAD_CE_WGM", 16)
  gptoss_shape = (M, V, K) == (16384, 128256, 2880)
  rocm72_requested = gptoss_shape and bool(getenv("GPTOSS_LMHEAD_CE_ROCM72", 0))
  rocm72_hipcc, rocm72_root = rocm_toolchain("/opt/rocm-7.2.1", rocm72_requested,
    flag="GPTOSS_LMHEAD_CE_ROCM72")
  rocm72 = rocm72_root is not None
  lds_pad = getenv("GPTOSS_LMHEAD_CE_LDS_PAD", 0) if gptoss_shape else 0
  assert lds_pad in (0, 2)
  store4 = gptoss_shape and rocm72 and lds_pad == 2 and wgm == 16 and bool(getenv("GPTOSS_LMHEAD_CE_STORE4", 0))
  store4_suffix = "_s4" if store4 else ""
  threads = UOp.special(512, "lidx0")
  workgroups = UOp.special((M // 256) * (V // 256), "gidx0")
  sink = UOp.sink(partial_max.base, partial_sum.base, partial_target.base, saved_logits.base,
                  hidden.base, weight.base, weight_extra.base, targets.base, threads, workgroups,
                  arg=KernelInfo(f"hk_lmhead_ce_saved_logits_w{wgm}_r72{int(rocm72)}p{lds_pad}{store4_suffix}_{M}_{V}_{K}",
                                 estimates=Estimates(ops=2*M*V*K + 3*M*V,
                                                     mem=(M*K+V*K+M*V)*2 + M*(V//256)*8 + M*4)))
  src = (pathlib.Path(__file__).parent.parent.parent/"thunder"/"amd"/"lmhead_ce_bf16.cpp").read_text()
  lib = HIPCCCompiler("gfx950", [f"-I{(pathlib.Path(__file__).parent.parent.parent/'thunder'/'amd'/'include').as_posix()}",
                                  "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math", "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
                                  f"-DGEMM_M={M}", f"-DGEMM_N={V}", f"-DGEMM_K={K}",
                                  f"-DLMHEAD_CE_WGM={wgm}", f"-DLMHEAD_CE_LDS_PAD={lds_pad}", f"-DLMHEAD_CE_STORE4={int(store4)}"],
                     hipcc_path=rocm72_hipcc,
                     rocm_path=rocm72_root).compile_cached(src)
  return hip_program(sink, src, lib)

@functools.cache
def _custom_lmhead_ce_mxfp8(scratch:UOp, hidden_q:UOp, weight_q:UOp, hidden_scales:UOp, weight_scales:UOp,
                            targets:UOp, *, dname:str) -> UOp:
  M, K = hidden_q.shape
  V, K2 = weight_q.shape
  assert K == K2 and M % 256 == V % 256 == K % 128 == 0
  assert hidden_scales.shape == (K // 128, M) and weight_scales.shape == (K // 128, V)
  assert scratch.shape == ((2*M*(V//256)+M)*4 + M*V*2,)
  threads = UOp.special(512, "lidx0")
  workgroups = UOp.special((M // 128) * (V // 256), "gidx0")
  sink = UOp.sink(scratch.base, hidden_q.base, weight_q.base, hidden_scales.base, weight_scales.base, targets.base,
                  threads, workgroups,
                  arg=KernelInfo(f"hk_lmhead_ce_mxfp8_saved_logits_{M}_{V}_{K}",
                                 estimates=Estimates(ops=2*M*V*K + 3*M*V,
                                                     mem=M*K + V*K + M*V*2 + (M+V)*(K//32) +
                                                         M*(V//256)*8 + M*4)))
  amd = pathlib.Path(__file__).parent.parent.parent/"thunder"/"amd"
  src = (amd/"lmhead_ce_mxfp8.cpp").read_text()
  lib = HIPCCCompiler("gfx950", [f"-I{(amd/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DGEMM_M={M}", f"-DGEMM_N={V}", f"-DGEMM_K={K}",
                                 "-DMX_TILE_M128=1", "-DMX_SINGLE_BUFFER=1", "-DMX_REG_PIPELINE=0"]).compile_cached(src)
  return hip_program(sink, src, lib)

@functools.cache
def _custom_lmhead_quantize_mxfp8_padded(q:UOp, si:UOp, x:UOp, *, dname:str) -> UOp:
  rows, H = x.shape
  rows2, K = q.shape
  assert rows == rows2 and H <= K and H % 32 == K % 128 == 0 and si.shape == (K // 128, rows)
  threads = UOp.special(256, "lidx0")
  workgroups = UOp.special((rows * (K // 128) + 255) // 256, "gidx0")
  sink = UOp.sink(q.base, si.base, x.base, threads, workgroups,
                  arg=KernelInfo(f"lmhead_quantize_mxfp8_padded_{rows}_{H}_{K}",
                                 estimates=Estimates(ops=rows*K, mem=rows*(H*2+K+K//32))))
  amd = pathlib.Path(__file__).parent.parent.parent/"thunder"/"amd"
  src = (amd/"lmhead_quantize_mxfp8_padded.cpp").read_text()
  lib = HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", f"-DROWS={rows}", f"-DH_DIM={H}", f"-DK_DIM={K}"]).compile_cached(src)
  return hip_program(sink, src, lib)

@functools.cache
def _custom_lmhead_ce_finalize_mxfp8(lse:UOp, loss_parts:UOp, scratch:UOp, *, dname:str, M:int, V:int) -> UOp:
  assert lse.shape == (M,) and loss_parts.shape == (M // 256,)
  threads, workgroups = UOp.special(256, "lidx0"), UOp.special(M // 256, "gidx0")
  sink = UOp.sink(lse.base, loss_parts.base, scratch.base, threads, workgroups,
                  arg=KernelInfo(f"lmhead_ce_finalize_mxfp8_{M}_{V}",
                                 estimates=Estimates(ops=4*M*(V//256), mem=2*M*(V//256)*4 + 2*M*4)))
  amd = pathlib.Path(__file__).parent.parent.parent/"thunder"/"amd"
  src = (amd/"lmhead_ce_finalize_mxfp8.cpp").read_text()
  lib = HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", f"-DGEMM_M={M}", f"-DGEMM_N={V}",
                                 f"-DMX_CE_EXACT_EXP={getenv('MX_CE_EXACT_EXP', 0)}"]).compile_cached(src)
  return hip_program(sink, src, lib)

@functools.cache
def _custom_lmhead_ce_finalize_bf16(lse:UOp, partial_max:UOp, partial_sum:UOp, *, dname:str, M:int, V:int) -> UOp:
  assert (M, V) == (16384, 128256) and partial_max.shape == partial_sum.shape == (M, V // 256)
  assert lse.shape == (M,) and lse.dtype == partial_max.dtype == partial_sum.dtype == dtypes.float32
  threads = 16
  workgroups = M // threads
  sink = UOp.sink(lse.base, partial_max.base, partial_sum.base,
                  UOp.special(threads, "lidx0"), UOp.special(workgroups, "gidx0"),
                  arg=KernelInfo(f"gptoss_lmhead_ce_finalize_bf16_t{threads}_{M}_{V}",
                                 estimates=Estimates(ops=5*M*(V//256), mem=(2*M*(V//256)+M)*4)))
  amd = pathlib.Path(__file__).parent.parent.parent/"thunder"/"amd"
  src = (amd/"lmhead_ce_finalize_bf16.cpp").read_text()
  lib = HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", f"-DGEMM_M={M}", f"-DGEMM_N={V}",
                                 f"-DFINALIZE_THREADS={threads}"]).compile_cached(src)
  return hip_program(sink, src, lib)

def _alloc(shape:tuple[int, ...], dtype, ref:Tensor) -> Tensor:
  axis = ref.uop.axis if isinstance(ref.device, tuple) else None
  return alloc_like(shape, dtype, ref.device, axis)

def _lmhead_quantize_mxfp8_padded(x:Tensor, K:int) -> tuple[Tensor, Tensor]:
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE
  rows, H = x.shape
  q = _alloc((rows, K), FP8_DTYPE, x)
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  si = alloc_like((K // 128, rows), dtypes.uint32, x.device, None if axis is None else (1 if axis == 0 else 0))
  q, si, *_ = Tensor.custom_kernel(q, si, x,
    fxn=functools.partial(_custom_lmhead_quantize_mxfp8_padded, dname=dname_of(x.device)))
  return q, si

@functools.cache
def _fwd_fxn(hidden_p:UOp, weight_p:UOp, targets_p:UOp, device) -> tuple[Tensor, ...]:
  hidden, weight, targets = Tensor(hidden_p, device=device), Tensor(weight_p, device=device), Tensor(targets_p, device=device)
  logical_lead = hidden.shape[:-1]
  mx_enabled = bool(getenv("LINEAR_CE_MX_FWD", 0))
  mx_multi = mx_enabled and isinstance(device, tuple)
  if mx_multi:
    # Execute the fused kernel on each DP device's physical token shard, with replicated vocabulary weights.
    def physical_param(x:Tensor) -> Tensor:
      assert x.uop.op is Ops.UNSHARD and x.uop.axis == 0
      return Tensor(x.uop.src[0])
    hidden, targets = physical_param(hidden), physical_param(targets)
    assert weight.uop.axis is None
  lead, H, V = hidden.shape[:-1], hidden.shape[-1], weight.shape[0]
  no_pad_bf16 = not mx_enabled and bool(getenv("LINEAR_CE_BF16_NO_PAD", 0)) and H % 64 == 0
  M, K = math.prod(lead), H if no_pad_bf16 else math.ceil(H / 256) * 256
  h = hidden.reshape(M, H).pad(((0, 0), (0, K-H)))
  w = weight.pad(((0, 0), (0, K-H)))
  pshape = (M, V // 256)
  # Default off: MXFP8 changes the forward logits and therefore the loss/gradients slightly. Keep the BF16
  # checkpoint because the current backward paths consume it, but never materialize a second logits tensor.
  if mx_enabled:
    hq, hsi = _lmhead_quantize_mxfp8_padded(hidden.reshape(M, H).contiguous(), K)
    wq, wsi = _lmhead_quantize_mxfp8_padded(weight.contiguous(), K)
    partial_elems = 2 * M * (V // 256) + M
    stat_bytes, logits_bytes = partial_elems * 4, M * V * 2
    scratch = _alloc((stat_bytes + logits_bytes,), dtypes.uint8, hidden)
    scratch, *_ = Tensor.custom_kernel(scratch, hq, wq, hsi, wsi, targets.reshape(M).cast(dtypes.int32),
      fxn=functools.partial(_custom_lmhead_ce_mxfp8, dname=dname_of(device)))
    stats = scratch[:stat_bytes].bitcast(dtypes.float32)
    pn = M * (V // 256)
    pmax, psum = stats[:pn].reshape(pshape), stats[pn:2*pn].reshape(pshape)
    ptarget = stats[2*pn:2*pn+M]
    saved_logits = scratch[stat_bytes:].bitcast(dtypes.bfloat16).reshape(M, V)
  else:
    pmax, psum = (_alloc(pshape, dtypes.float32, hidden) for _ in range(2))
    ptarget = _alloc((M,), dtypes.float32, hidden)
    saved_logits = _alloc((M, V), dtypes.bfloat16, hidden)
    pmax, psum, ptarget, saved_logits, *_ = Tensor.custom_kernel(
      pmax, psum, ptarget, saved_logits, h, w, w, targets.reshape(M).cast(dtypes.int32),
      fxn=functools.partial(_custom_lmhead_ce, dname=dname_of(device)))
  if mx_enabled:
    lse, loss_parts = _alloc((M,), dtypes.float32, hidden), _alloc((M // 256,), dtypes.float32, hidden)
    lse, loss_parts, *_ = Tensor.custom_kernel(lse, loss_parts, scratch,
      fxn=functools.partial(_custom_lmhead_ce_finalize_mxfp8, dname=dname_of(device), M=M, V=V))
    if mx_multi:
      ndev = len(device)
      lse = Tensor(lse.uop.unshard(0), device=device)
      saved_logits = Tensor(saved_logits.uop.unshard(0), device=device)
      loss_parts = Tensor(loss_parts.uop.unshard(0), device=device)
      loss = loss_parts.sum() / ndev
    else: loss = loss_parts.sum()
  else:
    # The GPT-OSS BF16 path has 501 vocabulary tiles. A dedicated finalize preserves the generated kernels'
    # scalar FP32 order while combining their max and log-sum-exp passes into one launch. It keeps the same
    # per-device row shard and physical output shape, so neither DP communication nor downstream layout changes.
    local_m = pmax.uop.shard_shape[0] if isinstance(device, tuple) and pmax.uop.axis is not None else M
    gptoss_finalize = (local_m, V, H) == (16384, 128256, 2880) and bool(getenv("GPTOSS_LMHEAD_CE_FINALIZE_BF16", 1))
    if gptoss_finalize:
      lse = _alloc((M,), dtypes.float32, hidden)
      lse, *_ = Tensor.custom_kernel(lse, pmax, psum,
        fxn=functools.partial(_custom_lmhead_ce_finalize_bf16, dname=dname_of(device), M=local_m, V=V))
    else:
      row_max = pmax.max(-1)
      lse = row_max + (psum * (pmax - row_max.reshape(M, 1)).exp()).sum(-1).log()
    loss = (lse - ptarget).mean()
  return loss, lse.reshape(*logical_lead), saved_logits


@functools.cache
def _custom_bwd_hk(dh_partial:UOp, dw_partial:UOp, hidden:UOp, weight:UOp, targets:UOp, lse:UOp, scale:UOp,
                   *, dname:str, part_n:int, part_m:int) -> UOp:
  M, K = hidden.shape
  V, K2 = weight.shape
  assert K == K2 and M % 128 == V % 128 == K % 32 == 0
  threads = UOp.special(512, "lidx0")
  workgroups = UOp.special(part_n * part_m, "gidx0")
  sink = UOp.sink(dh_partial.base, dw_partial.base, hidden.base, weight.base, targets.base, lse.base, scale.base,
                  threads, workgroups,
                  arg=KernelInfo(f"lmhead_ce_bwd_hk_{M}_{V}_{K}_{part_n}_{part_m}",
                                 estimates=Estimates(ops=6*M*V*K, mem=(M*K*part_n+V*K*part_m)*8)))
  src = (pathlib.Path(__file__).parent.parent.parent/"thunder"/"amd"/"lmhead_ce_bwd_hk.cpp").read_text()
  inc = (pathlib.Path(__file__).parent.parent.parent/"thunder"/"amd"/"include").as_posix()
  lib = HIPCCCompiler("gfx950", [f"-I{inc}", "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DTOKENS={M}", f"-DVOCAB={V}", f"-DHIDDEN={K}",
                                 f"-DPART_N={part_n}", f"-DPART_M={part_m}"]).compile_cached(src)
  return hip_program(sink, src, lib)

@functools.cache
def _custom_bwd_persistent(dh:UOp, dw:UOp, dlogits:UOp, barrier:UOp, hidden:UOp, weight:UOp,
                           targets:UOp, lse:UOp, scale:UOp, *, dname:str) -> UOp:
  M, K = hidden.shape
  V, K2 = weight.shape
  assert K == K2 and dh.shape == (M, K) and dw.shape[-2:] == (V, K) and dlogits.shape == (M, V)
  assert M % 256 == V % 256 == K % 256 == 0 and barrier.shape == (2,)
  threads, workgroups = UOp.special(512, "lidx0"), UOp.special(256, "gidx0")
  sink = UOp.sink(dh.base, dw.base, dlogits.base, barrier.base, hidden.base, weight.base,
                  targets.base, lse.base, scale.base, threads, workgroups,
                  arg=KernelInfo(f"lmhead_ce_bwd_persistent_{M}_{V}_{K}",
                                 estimates=Estimates(ops=6*M*V*K, mem=(M*K+V*K+3*M*V)*2)))
  amd = pathlib.Path(__file__).parent.parent.parent/"thunder"/"amd"
  src = (amd/"lmhead_ce_bwd_persistent.cpp").read_text()
  lib = HIPCCCompiler("gfx950", [f"-I{amd.as_posix()}", f"-I{(amd/'include').as_posix()}",
                                 "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DTOKENS={M}", f"-DVOCAB={V}", f"-DHIDDEN={K}",
                                 "-DPERSIST_WG=256"]).compile_cached(src)
  return hip_program(sink, src, lib)


@functools.cache
def _custom_ce_dual_mx(qrow:UOp, sirow:UOp, qcol:UOp, sicol:UOp,
                       logits:UOp, lse:UOp, targets:UOp, scale:UOp,
                       *, dname:str) -> UOp:
  M, V = logits.shape
  assert qrow.shape == (M, V) and qcol.shape == (V, M) and M % 32 == 0 and V % 256 == 0
  assert sirow.shape == (V//128, M) and sicol.shape == (M//128, V) and M % 128 == V % 128 == 0
  # GPT-OSS' 128256 vocabulary is exactly 167 x 768 columns. Keeping the per-wave work unchanged while
  # grouping twelve waves per workgroup reduced this producer by 9.9-10.2% with the production ROCm 7.1
  # toolchain (ABBA64, GPUs 2/3), with byte-exact poisoned full outputs, 65 SGPR, 33 VGPR, and no spill.
  gptoss_tile768 = (M, V) == (16384, 128256) and getenv("GPTOSS_CE_DUAL_TILE768", 1)
  tile_n = 768 if gptoss_tile768 else 256
  # This exact-shape GPT-OSS path packs each row's eight FP8 dwords into one aligned 32-byte vector store.
  # It keeps generic/Llama shapes unchanged while reducing this producer by ~0.6% on both tested MI350Xs.
  vec_row_store = gptoss_tile768 and getenv("GPTOSS_CE_DUAL_VEC_ROW_STORE", 1)
  # A wider bank rotation for the exact 768-column tile improves both row and transpose reads. Full-buffer
  # byte-exact/repeat ABBA on GPUs 0/5 and production DP2/L2 both favored eight pad slots per 32 columns.
  lds_block_pad = 8 if gptoss_tile768 and getenv("GPTOSS_CE_DUAL_LDS_BLOCK_PAD8", 1) else 1
  assert V % tile_n == 0
  threads = UOp.special(tile_n, "lidx0")
  workgroups = UOp.special((M//32) * (V//tile_n), "gidx0")
  sink = UOp.sink(qrow.base, sirow.base, qcol.base, sicol.base, logits.base, lse.base, targets.base, scale.base,
                  threads, workgroups,
                  arg=KernelInfo(f"ce_dlogits_dual_mxfp8_{M}_{V}_t{tile_n}_v{vec_row_store}_b{lds_block_pad}",
                                 estimates=Estimates(ops=5*M*V, mem=4*M*V+M*V//32)))
  src = (pathlib.Path(__file__).parent/"ce_dlogits_dual_mxfp8.cpp").read_text()
  lib = HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", f"-DM_DIM={M}", f"-DN_DIM={V}",
                                 f"-DTHREADS_PER_WG={tile_n}", f"-DNATIVE_MXFP8_CVT={getenv('NATIVE_MXFP8_CVT', 0)}",
                                 f"-DVEC_ROW_STORE={vec_row_store}", f"-DLDS_BLOCK_PAD={lds_block_pad}"]).compile_cached(src)
  return hip_program(sink, src, lib)

def _ce_dual_mx(logits:Tensor, lse:Tensor, targets:Tensor, scale:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE
  M, V = logits.shape
  axis = logits.uop.axis if isinstance(logits.device, tuple) else None
  qrow = alloc_like((M, V), FP8_DTYPE, logits.device, axis)
  col_axis = None if axis is None else (1 if axis == 0 else 0)
  sirow = alloc_like((V//128, M), dtypes.uint32, logits.device, col_axis)
  qcol = alloc_like((V, M), FP8_DTYPE, logits.device, col_axis)
  sicol = alloc_like((M//128, V), dtypes.uint32, logits.device, axis)
  qrow, sirow, qcol, sicol, *_ = Tensor.custom_kernel(
    qrow, sirow, qcol, sicol, logits, lse, targets.cast(dtypes.int32), scale,
    fxn=functools.partial(_custom_ce_dual_mx, dname=dname_of(logits.device)))
  return qrow, sirow, qcol, sicol

@functools.cache
def _custom_bwd_dlogits(dlogits:UOp, hidden:UOp, weight:UOp, weight_extra:UOp, targets:UOp, lse:UOp, scale:UOp,
                        *, dname:str) -> UOp:
  M, K = hidden.shape
  V, K2 = weight.shape
  assert K == K2 and dlogits.shape == (M, V) and M % 256 == V % 256 == K % 64 == 0
  threads = UOp.special(512, "lidx0")
  workgroups = UOp.special((M // 256) * (V // 256), "gidx0")
  sink = UOp.sink(dlogits.base, hidden.base, weight.base, weight_extra.base, targets.base, lse.base, scale.base,
                  threads, workgroups,
                  arg=KernelInfo(f"hk_lmhead_ce_dlogits_{M}_{V}_{K}",
                                 estimates=Estimates(ops=2*M*V*K + 3*M*V, mem=(M*K+V*K+M*V)*2)))
  amd = pathlib.Path(__file__).parent.parent.parent/"thunder"/"amd"
  src = (amd/"lmhead_ce_dlogits.cpp").read_text()
  lib = HIPCCCompiler("gfx950", [f"-I{(amd/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4", "-ffast-math",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DGEMM_M={M}", f"-DGEMM_N={V}", f"-DGEMM_K={K}"]).compile_cached(src)
  return hip_program(sink, src, lib)


@functools.cache
def _custom_reduce_dh(out:UOp, partials:UOp, *, dname:str, part_n:int) -> UOp:
  M, K = out.shape
  threads = UOp.special(256, "lidx0")
  groups = UOp.special((M*K + 255)//256, "gidx0")
  sink = UOp.sink(out.base, partials.base, threads, groups,
                  arg=KernelInfo(f"lmhead_ce_reduce_dh_{M}_{K}_{part_n}",
                                 estimates=Estimates(ops=(part_n-1)*M*K, mem=(part_n+1)*M*K*4)))
  src = (pathlib.Path(__file__).parent.parent.parent/"thunder"/"amd"/"lmhead_ce_reduce_dh.cpp").read_text()
  lib = HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", f"-DTOKENS={M}", f"-DHIDDEN={K}",
                                 f"-DPART_N={part_n}"]).compile_cached(src)
  return hip_program(sink, src, lib)

@functools.cache
def _custom_reduce_dw(out:UOp, partials:UOp, *, dname:str, part_m:int) -> UOp:
  V, K = out.shape[-2:]
  threads = UOp.special(256, "lidx0")
  groups = UOp.special((V*K + 255)//256, "gidx0")
  sink = UOp.sink(out.base, partials.base, threads, groups,
                  arg=KernelInfo(f"lmhead_ce_reduce_dw_{V}_{K}_{part_m}",
                                 estimates=Estimates(ops=(part_m-1)*V*K, mem=(part_m+1)*V*K*4)))
  src = (pathlib.Path(__file__).parent.parent.parent/"thunder"/"amd"/"lmhead_ce_reduce_dw.cpp").read_text()
  lib = HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", f"-DVOCAB={V}", f"-DHIDDEN={K}",
                                 f"-DPART_M={part_m}"]).compile_cached(src)
  return hip_program(sink, src, lib)

def _backward(gradient:UOp, call:UOp) -> tuple:
  from extra.gemm.cdna_asm_gemm import asm_gemm, hk_bf16_atb_gemm
  from extra.llama_kernels.fused_ce import _custom_fused_ce_loss_bwd
  hidden_u, weight_u, targets_u = call.src[1:4]
  device = hidden_u.device
  hidden, weight = Tensor(hidden_u, device=device), Tensor(weight_u, device=device)
  targets = Tensor(targets_u, device=device)
  lead, H, V = hidden.shape[:-1], hidden.shape[-1], weight.shape[0]
  M, K = math.prod(lead), math.ceil(H / 256) * 256
  hp = hidden.reshape(M, H).pad(((0, 0), (0, K-H)))
  wp = weight.pad(((0, 0), (0, K-H)))
  lse = Tensor(call.unbound_outputs[1], device=device).reshape(M)
  saved_logits = Tensor(call.unbound_outputs[2], device=device).reshape(M, V)
  scale = (Tensor(gradient, device=device).float().reshape(1) / M).contiguous()
  # Experimental one-launch producer/barrier/owner pipeline, specific to the 256-CU gfx950 target.
  if getenv("LINEAR_CE_PERSISTENT_BWD", 0):
    if isinstance(device, tuple):
      local_m = M // len(device)
      dh = Tensor(Tensor.invalids(local_m, K, dtype=dtypes.bfloat16, device=device).uop.unshard(0), device=device)
      dw_local = Tensor(Tensor.invalids(1, V, K, dtype=dtypes.bfloat16, device=device).uop.unshard(0), device=device)
      dlogits = Tensor(Tensor.invalids(local_m, V, dtype=dtypes.bfloat16, device=device).uop.unshard(0), device=device)
    else:
      dh = Tensor.invalids(M, K, dtype=dtypes.bfloat16, device=device)
      dw_local = Tensor.invalids(V, K, dtype=dtypes.bfloat16, device=device)
      dlogits = Tensor.invalids(M, V, dtype=dtypes.bfloat16, device=device)
    barrier = Tensor.zeros(2, dtype=dtypes.int32, device=device).contiguous()
    dh, dw_local, *_ = Tensor.custom_kernel(
      dh, dw_local, dlogits, barrier, hp, wp, targets.reshape(M).cast(dtypes.int32), lse, scale,
      fxn=functools.partial(_custom_bwd_persistent, dname=dname_of(device)))
    dw = dw_local.sum(0) if dw_local.ndim == 3 else dw_local
    return dh[:, :H].reshape(*lead, H).uop, dw[:, :H].uop, None

  # Experimental no-dlogits path; its FP32 output partitions are bandwidth-bound on MI350X.
  if getenv("LINEAR_CE_HK_BWD", 0):
    ndev = len(device) if isinstance(device, tuple) else 1
    local_m = M // ndev
    part_n, part_m = min(PART_N, V//128), min(PART_M, local_m//128)
    if isinstance(device, tuple):
      dhp = Tensor(Tensor.invalids(part_n, local_m, K, dtype=dtypes.float32, device=device).uop.unshard(1), device=device)
      dwp = Tensor(Tensor.invalids(1, part_m, V, K, dtype=dtypes.float32, device=device).uop.unshard(0), device=device)
    else:
      dhp = Tensor.invalids(part_n, M, K, dtype=dtypes.float32, device=device)
      dwp = Tensor.invalids(part_m, V, K, dtype=dtypes.float32, device=device)
    dhp, dwp, *_ = Tensor.custom_kernel(
      dhp, dwp, hp, wp, targets.reshape(M).cast(dtypes.int32), lse, scale,
      fxn=functools.partial(_custom_bwd_hk, dname=dname_of(device), part_n=part_n, part_m=part_m))
    if isinstance(device, tuple):
      dh = Tensor(Tensor.invalids(M//len(device), K, dtype=dtypes.bfloat16, device=device).uop.unshard(0), device=device)
      dw_local = Tensor(Tensor.invalids(1, V, K, dtype=dtypes.bfloat16, device=device).uop.unshard(0), device=device)
    else:
      dh = Tensor.invalids(M, K, dtype=dtypes.bfloat16, device=device)
      dw_local = Tensor.invalids(V, K, dtype=dtypes.bfloat16, device=device)
    dh, *_ = Tensor.custom_kernel(dh, dhp, fxn=functools.partial(_custom_reduce_dh, dname=dname_of(device), part_n=part_n))
    dw_local, *_ = Tensor.custom_kernel(dw_local, dwp,
                                         fxn=functools.partial(_custom_reduce_dw, dname=dname_of(device), part_m=part_m))
    dw = dw_local.sum(0) if dw_local.ndim == 3 else dw_local
    return dh[:, :H].reshape(*lead, H).uop, dw[:, :H].uop, None

  # Directly convert saved logits to both rowwise and columnwise MXFP8 dLogits. This avoids the 4 GiB BF16
  # dLogits intermediate and feeds both GEMMs with already-quantized operands.
  if getenv("LINEAR_CE_DUAL_MX_BWD", 0):
    from extra.llama_kernels.transpose_quantize_mxfp8 import transpose_quantize_mxfp8, transpose_quantize_mxfp8_padded
    logits = saved_logits.reshape(M, V)
    dlq, dlsi, dlcol, dlcolsi = _ce_dual_mx(logits, lse, targets.reshape(M), scale)
    if getenv("LINEAR_CE_PADDED_TQ", 0):
      wcol, wcole8, wcolsi = transpose_quantize_mxfp8_padded(weight, K)
      hcol, hcole8, hcolsi = transpose_quantize_mxfp8_padded(hidden.reshape(M, H), K)
    else:
      wcol, wcole8, wcolsi = transpose_quantize_mxfp8(wp.contiguous())
      hcol, hcole8, hcolsi = transpose_quantize_mxfp8(hp.contiguous())
    # gemm_mxfp8 only reads packed scales; its raw-e8 arguments exist for the generic grad_fxn. This path is
    # already inside the terminal fused-loss backward, so alias the packed buffers into those unused slots.
    # This is the GPT-OSS LM-head dHidden GEMM. Mark the callsite explicitly: the generic MX backward
    # callback is not involved in the fused-CE path, so shape detection there cannot select this kernel.
    # asm_gemm retains the exact local (M,N,K) guard before choosing the dedicated binary.
    dh = asm_gemm(dlq, wcol.T, mx=True, mx_scales=(dlsi, dlsi, wcolsi, wcole8),
                  gptoss_lmhead_dh=True)[:, :H].reshape(*lead, H)
    shard_dw = isinstance(device, tuple) and getenv("ZERO_OPTIM", 0) and getenv("ZERO2", 0) and getenv("GPTOSS_ZERO2_LMHEAD", 0)
    dw = asm_gemm(dlcol, hcol.T, mx=True, mx_scales=(dlcolsi, dlcolsi, hcolsi, hcole8),
                  gptoss_lmhead_reduce_scatter=bool(shard_dw))[:, :H]
    return dh.uop, dw.uop, None

  # Retain the recompute path for kernel comparisons. The default reuses forward's BF16 logits and runs tuned
  # dHidden/dWeight GEMMs.
  if getenv("FUSED_LINEAR_CE_DLOGITS", 0):
    if isinstance(device, tuple):
      local_m = M // len(device)
      dlogits = Tensor(Tensor.invalids(local_m, V, dtype=dtypes.bfloat16, device=device).uop.unshard(0), device=device)
    else:
      dlogits = Tensor.invalids(M, V, dtype=dtypes.bfloat16, device=device)
    dlogits, *_ = Tensor.custom_kernel(
      dlogits, hp, wp, wp, targets.reshape(M).cast(dtypes.int32), lse, scale,
      fxn=functools.partial(_custom_bwd_dlogits, dname=dname_of(device)))
  else:
    logits = saved_logits.reshape(*(lead if len(lead) > 1 else (1, M)), V)
    if isinstance(device, tuple):
      axis, ndev = logits.uop.axis, len(device)
      local_shape = tuple(s//ndev if i == axis else s for i,s in enumerate(logits.shape))
      dlogits = Tensor(Tensor.invalids(*local_shape, dtype=dtypes.bfloat16, device=device).uop.unshard(axis), device=device)
      rows_per_dev, seq_per_dev = local_shape[0] * local_shape[1], local_shape[1]
    else:
      dlogits = Tensor.invalids(*logits.shape, dtype=dtypes.bfloat16, device=device)
      rows_per_dev, seq_per_dev = M, lead[-1]
    dlogits, *_ = Tensor.custom_kernel(
      dlogits, logits, lse, targets.reshape(M).cast(dtypes.int32), scale,
      fxn=functools.partial(_custom_fused_ce_loss_bwd, vocab=V, rows=rows_per_dev,
                            seq=seq_per_dev, label_smoothing=0.0))
  dl = dlogits.reshape(M, V)
  if getenv("LINEAR_CE_MX_BWD", 0):
    dh = asm_gemm(dl, wp, mx=True)[:, :H].reshape(*lead, H)
    dw = asm_gemm(dl.T, hp, mx=True, a_pretranspose=dl)[:, :H]
  else:
    dh = asm_gemm(dl, wp)[:, :H].reshape(*lead, H)
    dw = hk_bf16_atb_gemm(hp.reshape(1, M, K), dl.reshape(1, M, V)).T[:, :H]
  return dh.uop, dw.uop, None

def fused_linear_cross_entropy(hidden:Tensor, weight:Tensor, targets:Tensor) -> Tensor:
  assert hidden.dtype == weight.dtype == dtypes.bfloat16 and hidden.shape[-1] == weight.shape[-1]
  fxn = _fwd_fxn(hidden.as_param(0).uop, weight.as_param(1).uop, targets.as_param(2).uop, hidden.device)
  outputs = UOp.call_with_outputs(tuple(x.uop for x in fxn), hidden.uop, weight.uop, targets.uop, grad_fxn=_backward)
  return Tensor(outputs[0])
