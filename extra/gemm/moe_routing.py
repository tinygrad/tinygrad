import functools, math, pathlib
from tinygrad import Tensor, dtypes, function
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from extra.hipcc import HIPCCCompiler
from extra.llama_kernels import alloc_like, owned_empty, dname_of

BLOCK_ROW = 256
DISPATCH_GATHER = getenv("DISPATCH_GATHER", 0)
FUSED_DISPATCH_COLW = getenv("FUSED_DISPATCH_COLW", 0)
GPTOSS_DISPATCH_THREADS = getenv("GPTOSS_DISPATCH_THREADS", 128)
assert GPTOSS_DISPATCH_THREADS in (128, 256), "GPTOSS_DISPATCH_THREADS must be 128 or 256"
GGATHER_SUM_HIP = getenv("GGATHER_SUM_HIP", 0)
GGATHER_SUM_THREADS = getenv("GGATHER_SUM_THREADS", 64)
assert GGATHER_SUM_THREADS in (64, 128, 192, 256), "GGATHER_SUM_THREADS must be 64, 128, 192, or 256"
# BEAM_GLUE=1 -> custom glue kernels use opts_to_apply=None (beam-searches them toward hw max) instead of () (naive).
# GLUE_BEAM sets a PER-KERNEL beam depth (KernelInfo.beam, honored by codegen/__init__.py:310) so in-pipeline they
# beam at the SAFE depth 2 (the isolated 4.5x depth) instead of the global JITBEAM=3 (which finds a GPU-hanging opt).
GLUE_OPTS = None if getenv("BEAM_GLUE", 0) else ()
GLUE_BEAM = getenv("BEAM_GLUE_DEPTH", 2) if getenv("BEAM_GLUE", 0) else 0

ROUTER_FP32_WGRAD = getenv("ROUTER_FP32_WGRAD", 0)
ROUTER_FP32_WGRAD_WARPS = getenv("ROUTER_FP32_WGRAD_WARPS", 16)
ROUTER_FP32_WGRAD_ACCUMS = getenv("ROUTER_FP32_WGRAD_ACCUMS", 1)
ROUTER_FP32_WGRAD_DUAL_EXPERT = getenv("ROUTER_FP32_WGRAD_DUAL_EXPERT", 0)
from extra.gptoss_kernels.router_topk import _router_bias_grad_lookup

@functools.cache
def _router_mfma_fwd(out:UOp, x:UOp, weight:UOp, bias:UOp, *, dname:str) -> UOp:
  *lead, K = x.shape
  M = math.prod(lead)
  E = weight.shape[0]
  threads = UOp.special(256, "lidx0")
  workgroups = UOp.special((M + 63) // 64, "gidx0")
  zero = UOp.const(0, dtypes.int32)
  accesses = (out.index(zero).store(UOp.const(0, out.dtype)), x.index(zero).load(), weight.index(zero).load(), bias.index(zero).load())
  sink = UOp.sink(*accesses, threads, workgroups,
                  arg=KernelInfo(f"moe_router_mfma_{M}_{K}_{E}",
                                 estimates=Estimates(ops=2*M*E*K, mem=(M*K+E*K+E)*2+M*E*4)))
  amd = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (amd/"moe_router_mfma.cpp").read_text()
  lib = HIPCCCompiler("gfx950", [f"-I{(amd/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DROUTER_M={M}", f"-DROUTER_K={K}",
                                 f"-DROUTER_E={E}"]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=lib)))

@functools.cache
def _router_fp32_wgrad_kernel(out:UOp, x:UOp, gradient:UOp, *, dname:str) -> UOp:
  K, E = x.shape[-1], gradient.shape[-1]
  M = math.prod(x.shape[:-1])
  assert out.shape == (E, K) and E == 32 and K % 16 == 0 and M % (4 * ROUTER_FP32_WGRAD_WARPS) == 0
  assert ROUTER_FP32_WGRAD_ACCUMS > 0 and ROUTER_FP32_WGRAD_ACCUMS & (ROUTER_FP32_WGRAD_ACCUMS - 1) == 0
  prefetch = 8 if (getenv("GPTOSS_ROUTER_WGRAD_PREFETCH", 0) and (M, K, E) == (16384, 2880, 32)
                  and ROUTER_FP32_WGRAD_WARPS == 16 and ROUTER_FP32_WGRAD_ACCUMS == 1 and ROUTER_FP32_WGRAD_DUAL_EXPERT) else 0
  threads = UOp.special(ROUTER_FP32_WGRAD_WARPS * 64, "lidx0")
  workgroups = UOp.special((K // 16) * (1 if ROUTER_FP32_WGRAD_DUAL_EXPERT else E // 16), "gidx0")
  sink = UOp.sink(out.base, x.base, gradient.base, threads, workgroups,
                  arg=KernelInfo(f"moe_router_fp32_wgrad_{M}_{K}_{E}_w{ROUTER_FP32_WGRAD_WARPS}_a{ROUTER_FP32_WGRAD_ACCUMS}"
                                 f"_de{int(ROUTER_FP32_WGRAD_DUAL_EXPERT)}"+("_pf8" if prefetch else ""),
                                 estimates=Estimates(ops=2*M*E*K, mem=M*(K*2+E*4)+E*K*2)))
  amd = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (amd/"moe_router_fp32_wgrad.cpp").read_text()
  lib = HIPCCCompiler("gfx950", [f"-I{(amd/'include').as_posix()}", "-std=c++20", "-DKITTENS_CDNA4",
                                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS", f"-DROUTER_M={M}", f"-DROUTER_K={K}",
                                 f"-DROUTER_E={E}", f"-DNUM_WARPS={ROUTER_FP32_WGRAD_WARPS}",
                                 f"-DNUM_ACCUMS={ROUTER_FP32_WGRAD_ACCUMS}",
                                 f"-DDUAL_EXPERT={int(ROUTER_FP32_WGRAD_DUAL_EXPERT)}", f"-DPREFETCH={prefetch}"]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=lib)))

def _router_fp32_wgrad(x:Tensor, gradient:Tensor) -> Tensor:
  E, K = gradient.shape[-1], x.shape[-1]
  # The router weight is replicated, so every device produces the full local-data contribution (E,K).
  out = Tensor.invalids(E, K, dtype=dtypes.bfloat16, device=x.device)
  return Tensor.custom_kernel(out, x, gradient,
    fxn=functools.partial(_router_fp32_wgrad_kernel, dname=str(x.device)))[0]

@functools.cache
def _router_dgrad_fused_kernel(out:UOp, weight:UOp, gradient:UOp, residual:UOp, e8:UOp, *, dname:str) -> UOp:
  M, K = out.shape
  E = weight.shape[0]
  assert (M, K, E) == (16384, 2880, 32)
  assert weight.shape == (E, K) and gradient.shape == (M, E)
  assert residual.shape == (M, 3072) and e8.shape == (M, 96)
  threads_x, threads_y = UOp.special(16, "lidx0"), UOp.special(4, "lidx1")
  workgroups_x, workgroups_y = UOp.special(K // 320, "gidx0"), UOp.special(M // 8, "gidx1")
  zero = UOp.const(0, dtypes.int32)
  accesses = (out.base.index(zero).store(UOp.const(0, out.dtype)),
              *(u.base.index(zero).load() for u in (weight, gradient, residual, e8)))
  sink = UOp.sink(*accesses,
                  threads_x, threads_y, workgroups_x, workgroups_y,
                  arg=KernelInfo("moe_router_dgrad_fused",
                                 estimates=Estimates(ops=2*M*K*E, mem=M*K*4+M*E*4+E*K*2+M*(K*2+K//32))))
  amd = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (amd/"moe_router_dgrad_fused.cpp").read_text()
  # This source has no HipKittens dependency. COMGR matches the compiler used by the generic reduction whose exact
  # FP32 association it reproduces, and also makes this small glue kernel independent of the host ROCm toolchain.
  lib = HIPCompiler("gfx950").compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=lib)))

def _router_dgrad_fused(weight:Tensor, gradient:Tensor, residual:Tensor, e8:Tensor, shape:tuple[int, ...]) -> Tensor:
  out = owned_empty(_sharded_invalids(shape, dtypes.bfloat16, gradient.device))
  if getenv("GPTOSS_ROUTER_GATHER", 0):
    from extra.gptoss_kernels.router_gather import gathered_router
    if (result:=gathered_router(out, weight, gradient, residual, e8, shape)) is not None: return result
  return Tensor.custom_kernel(out, weight, gradient, residual, e8,
    fxn=functools.partial(_router_dgrad_fused_kernel, dname=str(gradient.device)))[0]

def _router_mfma_bwd(gradient:UOp, kernel:UOp) -> tuple:
  _, x_u, weight_u, bias_u = kernel.src[1:5]
  x, weight, bias = (Tensor(u, device=u.device) for u in (x_u, weight_u, bias_u))
  fused_bias_u = _router_bias_grad_lookup(gradient)
  if ROUTER_FP32_WGRAD:
    g = Tensor(gradient, device=x_u.device)
    grad_weight = _router_fp32_wgrad(x, g)
    reference = x.float() @ weight.float().T + bias.float()
    grad_x, _, grad_bias_ref = reference.gradient(x, weight, bias, gradient=g)
    grad_bias = Tensor(fused_bias_u, device=bias.device).cast(bias.dtype) if fused_bias_u is not None else grad_bias_ref
    return None, grad_x.uop, grad_weight.uop, grad_bias.uop
  if getenv("ROUTER_MFMA_WGRAD_GEMM", 0):
    from extra.gemm.cdna_asm_gemm import hk_bf16_atb_gemm
    g = Tensor(gradient, device=x_u.device)
    *lead, K = x.shape
    E = weight.shape[0]
    Kp, Ep = ((K + 255) // 256) * 256, ((E + 255) // 256) * 256
    xp = x.pad((None, None, (0, Kp-K)))
    gp = g.cast(dtypes.bfloat16).reshape(*lead, E).pad(((0, 0),) * len(lead) + ((0, Ep-E),))
    grad_weight = hk_bf16_atb_gemm(xp, gp).T[:E, :K]
    reference = x.float() @ weight.float().T + bias.float()
    grad_x, _, grad_bias_ref = reference.gradient(x, weight, bias, gradient=g)
    grad_bias = Tensor(fused_bias_u, device=bias.device).cast(bias.dtype) if fused_bias_u is not None else grad_bias_ref
    return None, grad_x.uop, grad_weight.uop, grad_bias.uop
  if getenv("ROUTER_MFMA_BWD_GEMM", 0):
    from extra.gemm.cdna_asm_gemm import asm_gemm, hk_bf16_atb_gemm
    *lead, K = x.shape
    E, K2 = weight.shape
    assert K == K2
    Kp, Ep = ((K + 255) // 256) * 256, ((E + 255) // 256) * 256
    gp = Tensor(gradient, device=x_u.device).cast(dtypes.bfloat16).reshape(-1, E).pad(((0, 0), (0, Ep - E)))
    wp = weight.pad(((0, Ep - E), (0, Kp - K)))
    grad_x = asm_gemm(gp, wp).reshape(*lead, Kp)[..., :K]
    xp = x.pad((None, None, (0, Kp - K)))
    gp3 = gp.reshape(*lead, Ep)
    grad_weight = hk_bf16_atb_gemm(xp, gp3).T[:E, :K]
    grad_bias_ref = Tensor(gradient, device=x_u.device).float().sum(axis=tuple(range(len(lead))))
    grad_bias = Tensor(fused_bias_u, device=bias.device).cast(bias.dtype) if fused_bias_u is not None else grad_bias_ref
    return None, grad_x.uop, grad_weight.uop, grad_bias.uop
  reference = x.float() @ weight.float().T + bias.float()
  grad_x, grad_weight, grad_bias_ref = reference.gradient(x, weight, bias, gradient=Tensor(gradient, device=x_u.device))
  grad_bias = Tensor(fused_bias_u, device=bias.device).cast(bias.dtype) if fused_bias_u is not None else grad_bias_ref
  return None, grad_x.uop, grad_weight.uop, grad_bias.uop

def router_mfma(x:Tensor, weight:Tensor, bias:Tensor) -> Tensor:
  assert x.ndim >= 2 and weight.ndim == 2 and bias.ndim == 1
  K = x.shape[-1]
  E = weight.shape[0]
  assert weight.shape == (E, K) and bias.shape == (E,)
  assert x.dtype == weight.dtype == bias.dtype == dtypes.bfloat16
  assert E == 32 and K % 64 == 0
  if isinstance(x.device, tuple):
    assert x.uop.axis == 0, f"router MFMA requires axis-0 sharding, got axis={x.uop.axis}"
    local_shape = x.uop.shard_shape
    assert local_shape[-1] == K and math.prod(local_shape[:-1]) % 64 == 0, f"unsupported local router shape {local_shape}"
  else:
    assert math.prod(x.shape[:-1]) % 64 == 0
  # GPT-OSS supplies a dense, row-major norm output viewed as flat tokens. Preserve its producer buffer;
  # an explicit contiguous here otherwise materializes an additional full activation for the router.
  x = x if getenv("GPTOSS_ROUTER_INPUT_VIEW", 0) else x.contiguous()
  weight, bias = weight.contiguous(), bias.contiguous()
  out = _sharded_invalids((*x.shape[:-1], E), dtypes.float32, x.device)
  out = out.clone()
  out, *_ = Tensor.custom_kernel(out, x, weight, bias,
    fxn=functools.partial(_router_mfma_fwd, dname=str(x.device)), grad_fxn=_router_mfma_bwd)
  return out

def _router_quantize_bwd(dlogits:UOp, dquantized:UOp, *, call:UOp) -> tuple:
  x, weight, bias = (Tensor(u) for u in call.src[1:4])
  g = Tensor(dlogits)
  # Both branches meet at this function boundary, independent of backward traversal order. Use the exact saved
  # scale output exposed by dispatch/run_layer, preserving the BF16 cast before each branch is added.
  dx = _router_dgrad_fused(weight, g, Tensor(dquantized), Tensor(call.unbound_outputs[2]), x.shape)
  dw = _router_fp32_wgrad(x, g)
  fused_bias = _router_bias_grad_lookup(dlogits)
  db = (Tensor(fused_bias) if fused_bias is not None else g.sum(axis=0)).cast(bias.dtype)
  return dx.uop, dw.uop, db.uop

@function(precompile=True, grad_fxn=_router_quantize_bwd)
def router_quantize(x:Tensor, weight:Tensor, bias:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  """GPT-OSS training router and dispatch quantizer, with one explicit input-gradient join."""
  from extra.llama_kernels.quantize_mxfp8_fused import quantize_mxfp8_fused_qe8
  assert x.uop.shard_shape == (16384, 2880) and weight.shape == (32, 2880) and bias.shape == (32,)
  assert ROUTER_FP32_WGRAD and FUSED_DISPATCH_COLW and getenv("FUSED_DISPATCH_QE8", 1) and getenv("SAVE_DISPATCH_E8", 0)
  logits = router_mfma(x, weight, bias)
  q, e8 = quantize_mxfp8_fused_qe8(x.pad(((0, 0), (0, 192))))
  return logits, q, e8

def _sharded_invalids(shape:tuple[int, ...], dtype, device) -> Tensor:
  return alloc_like(shape, dtype, device, axis=0)

def _atomic_add(device:str) -> str:
  return "__hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);" if device == "AMD" \
    else "__atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED);"

def _blk_for(D:int) -> int:
  blk = 64
  while D % blk: blk //= 2
  return blk

def _kv_ranges(G, N, D, BLK):
  g = UOp.range(G, 0)
  m = UOp.range(N, 1)
  jo = UOp.range(D // BLK, 2)
  ji = UOp.range(BLK, 3, AxisType.LOCAL)
  return g, m, jo * BLK + ji, jo, ji

def _ggather_fwd_kernel(out:UOp, table:UOp, idx:UOp) -> UOp:
  G, M, D = out.shape
  g, m, j, jo, ji = _kv_ranges(G, M, D, _blk_for(D))
  row = idx.index(g, m).cast(dtypes.weakint)
  val = table.index(g, row, j).load()
  return out.index(g, m, j).store(val).end(g, m, jo, ji).sink(
    arg=KernelInfo(name=f"ggather_fwd_{M}_{D}", opts_to_apply=GLUE_OPTS, beam=GLUE_BEAM))

def _ggather_zero_kernel(out:UOp) -> UOp:
  i = UOp.range(out.numel(), 0)
  return out.flatten().index(i).store(UOp.const(0.0, out.dtype)).end(i).sink(arg=KernelInfo(name="ggather_zero"))

def _sharded_zeros(shape:tuple[int, ...], dtype, device) -> Tensor:
  return Tensor.custom_kernel(_sharded_invalids(shape, dtype, device), fxn=_ggather_zero_kernel)[0]

def _ggather_bwd(gradient:UOp, kernel:UOp) -> tuple:
  _, table_u, idx_u = kernel.src[1:4]
  dev = table_u.device
  device = (dev[0] if isinstance(dev, tuple) else dev).split(":")[0]
  G, R, D = table_u.shape
  gt = _sharded_zeros((G, R, D), dtypes.float32, dev)
  go = Tensor(gradient, device=dev)
  atomic_str = _atomic_add(device)
  def _bwd_kernel(gtab:UOp, gout:UOp, idx:UOp) -> UOp:
    Gk, M, Dk = gout.shape
    g, m, j, jo, ji = _kv_ranges(Gk, M, Dk, _blk_for(Dk))
    row = idx.index(g, m).cast(dtypes.weakint)
    val = gout.index(g, m, j).load().cast(dtypes.float32)
    atomic = UOp(Ops.CUSTOM, src=(gtab.index(g, row, j), val), arg=(atomic_str, dtypes.void))
    return atomic.end(g, m, jo, ji).sink(arg=KernelInfo(name=f"ggather_bwd_{M}_{Dk}", opts_to_apply=()))
  grad_table = Tensor.custom_kernel(gt, go, Tensor(idx_u, device=dev), fxn=_bwd_kernel)[0]
  return (None, grad_table.cast(table_u.dtype).uop, None)

def grouped_gather_rows(table:Tensor, idx:Tensor, n_groups:int) -> Tensor:
  G, R, D = table.shape
  M = idx.shape[1]
  out = _sharded_invalids((G, M, D), table.dtype, table.device)
  return Tensor.custom_kernel(out, table, idx, fxn=_ggather_fwd_kernel, grad_fxn=_ggather_bwd)[0]

@functools.cache
def _ggather_sum_fwd_hip(out:UOp, table:UOp, idx:UOp) -> UOp:
  # GPTOSS inverse-dispatch backward. One workgroup owns a token, shares its four row indices, and walks the
  # contiguous hidden dimension. This avoids the generic kernel's three workgroups/token and generalized modulo
  # address calculations while preserving the physical producer shape consumed by the following backward kernel.
  G, T, D = out.shape
  M, k = table.shape[1], idx.shape[1] // T
  assert (G, T, D, M, k) == (1, 16384, 3072, 73728, 4)
  assert table.shape == (G, M, D) and idx.shape == (G, T * k)
  assert out.dtype == table.dtype == dtypes.bfloat16 and idx.dtype == dtypes.int32
  threads, workgroups = UOp.special(GGATHER_SUM_THREADS, "lidx0"), UOp.special(G * T, "gidx0")
  elems = G * T * D
  sink = UOp.sink(out.base, table.base, idx.base, threads, workgroups,
                  arg=KernelInfo(f"ggather_sum_fwd_hip_{T}_{k}_{D}_t{GGATHER_SUM_THREADS}",
                                 estimates=Estimates(ops=(k - 1) * elems,
                                                     mem=(k * elems + elems) * table.dtype.itemsize + idx.numel() * 4)))
  src = (pathlib.Path(__file__).parent.parent/"thunder"/"amd"/"moe_ggather_sum.cpp").read_text()
  lib = HIPCCCompiler("gfx950", ["-std=c++20", "-O3", f"-DG_DIM={G}", f"-DT_DIM={T}", f"-DM_DIM={M}",
                                 f"-DD_DIM={D}", f"-DK_DIM={k}", f"-DTHREADS={GGATHER_SUM_THREADS}"]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=lib)))

def _ggather_sum_fwd_kernel(out:UOp, table:UOp, idx:UOp, *, dname:str|None=None) -> UOp:
  # Fused inverse-dispatch backward: gather the k expert copies of each token and accumulate them before writing.
  # The unfused form materializes (G,T*k,D), then immediately rereads it for sum(k); this writes only (G,T,D).
  G, T, D = out.shape
  k = idx.shape[1] // T
  assert table.shape[0] == G and table.shape[2] == D and idx.shape == (G, T * k)
  if GGATHER_SUM_HIP and dname == "AMD" and out.shape == (1, 16384, 3072) and table.shape == (1, 73728, 3072) and \
     idx.shape == (1, 65536) and out.dtype == table.dtype == dtypes.bfloat16 and idx.dtype == dtypes.int32:
    return _ggather_sum_fwd_hip(out, table, idx)
  g, t, j, jo, ji = _kv_ranges(G, T, D, _blk_for(D))
  ki = UOp.range(k, 4, AxisType.REDUCE)
  row = idx.index(g, t * k + ki).cast(dtypes.weakint)
  val = table.index(g, row, j).load().cast(dtypes.float32)
  summed = val.reduce(ki, arg=(Ops.ADD, 0)).cast(out.dtype)
  return out.index(g, t, j).store(summed).end(g, t, jo, ji).sink(
    arg=KernelInfo(name=f"ggather_sum_fwd_{T}_{k}_{D}", opts_to_apply=GLUE_OPTS, beam=GLUE_BEAM))

def grouped_gather_sum_rows(table:Tensor, idx:Tensor, n_groups:int, k:int) -> Tensor:
  G, _, D = table.shape
  assert G == n_groups and idx.shape[0] == G
  # Keep the output shape explicit so this custom kernel is the physical producer consumed by the following
  # qscale/router-dgrad expression, with no reshape/contiguous boundary.
  assert idx.shape[1] % k == 0
  out = owned_empty(_sharded_invalids((G, idx.shape[1] // k, D), table.dtype, table.device))
  dev = table.device[0] if isinstance(table.device, tuple) else table.device
  dname = str(dev).split(":")[0]
  result = Tensor.custom_kernel(out, table, idx, fxn=functools.partial(_ggather_sum_fwd_kernel, dname=dname))[0]
  if getenv("GPTOSS_ROUTER_GATHER", 0):
    from extra.gptoss_kernels.router_gather import mark_native
    result = mark_native(result)
  return result

def _gscatter_fwd_kernel(out:UOp, src:UOp, idx:UOp) -> UOp:
  G, M, D = out.shape
  k = idx.shape[1] // src.shape[1]
  g, m, j, jo, ji = _kv_ranges(G, idx.shape[1], D, _blk_for(D))
  row = idx.index(g, m).cast(dtypes.weakint)
  val = src.index(g, (m // k).cast(dtypes.weakint), j).load()
  return out.index(g, row, j).store(val).end(g, m, jo, ji).sink(
    arg=KernelInfo(name=f"gscatter_fwd_{idx.shape[1]}_{D}", opts_to_apply=GLUE_OPTS, beam=GLUE_BEAM))

def _gscatter_bwd(gradient:UOp, kernel:UOp) -> tuple:
  _, src_u, idx_u = kernel.src[1:4]
  dev = src_u.device
  G, T_l, D = src_u.shape
  k = idx_u.shape[1] // T_l
  sel = grouped_gather_rows(Tensor(gradient, device=dev), Tensor(idx_u, device=dev), G)
  return (None, sel.reshape(G, T_l, k, D).sum(2).cast(src_u.dtype).uop, None)

def grouped_scatter_rows(src:Tensor, idx:Tensor, m_l:int) -> Tensor:
  G, T_l, D = src.shape
  zero = _sharded_zeros((G, m_l, D), src.dtype, src.device)
  return Tensor.custom_kernel(zero, src, idx, fxn=_gscatter_fwd_kernel, grad_fxn=_gscatter_bwd)[0]

def _gscatter_bwd_hi(gradient:UOp, kernel:UOp) -> tuple:
  # like _gscatter_bwd but accumulates the gathered grad in bf16 instead of src.dtype -- for an fp8 src the grad
  # must not be re-quantized to fp8 in the dispatch backward (the fp8 rows' grad flows on to quantize_mxfp8's STE).
  _, src_u, idx_u = kernel.src[1:4]
  dev = src_u.device
  G, T_l, D = src_u.shape
  k = idx_u.shape[1] // T_l
  sel = grouped_gather_rows(Tensor(gradient, device=dev), Tensor(idx_u, device=dev), G)
  # accumulate over the k copies in fp32 (like the bf16-dispatch backward, whose grad is already grad_xq*qscale
  # in fp32) so the sum-over-k is byte-exact; the trailing *qscale in quantize_mxfp8's STE backward is a per-token
  # power of 2 and commutes with this sum.
  return (None, sel.reshape(G, T_l, k, D).float().sum(2).cast(dtypes.bfloat16).uop, None)

def grouped_scatter_rows_hi(src:Tensor, idx:Tensor, m_l:int) -> Tensor:
  G, T_l, D = src.shape
  zero = _sharded_zeros((G, m_l, D), src.dtype, src.device)
  return Tensor.custom_kernel(zero, src, idx, fxn=_gscatter_fwd_kernel, grad_fxn=_gscatter_bwd_hi)[0]

# --- inverse permutation map (NVIDIA MoE-permute style): grouped_row -> expanded source i=t*k+j (SENTINEL -1 for pad) ---
def _neg1_fill_kernel(out:UOp) -> UOp:
  i = UOp.range(out.numel(), 0)
  return out.flatten().index(i).store(UOp.const(-1, out.dtype)).end(i).sink(arg=KernelInfo(name="moe_neg1", opts_to_apply=()))

@functools.cache
def _invmap_kernel(src_row:UOp, dest_row:UOp) -> UOp:
  # dest_row is an injection expanded_row i -> grouped row; invert it: src_row[dest_row[i]] = i (contention-free).
  G, Mk = dest_row.shape
  g = UOp.range(G, 0)
  i = UOp.range(Mk, 1)
  m = dest_row.index(g, i).load().cast(dtypes.weakint)
  return src_row.index(g, m).store(i.cast(src_row.dtype)).end(i, g).sink(arg=KernelInfo(name=f"moe_invmap_{Mk}", opts_to_apply=()))

def build_src_row(dest_row:Tensor, m_l:int) -> Tensor:
  buf = owned_empty(_sharded_invalids((dest_row.shape[0], m_l), dtypes.int32, dest_row.device))
  buf = Tensor.custom_kernel(buf, fxn=_neg1_fill_kernel)[0]
  return Tensor.custom_kernel(buf, dest_row, fxn=_invmap_kernel)[0]

@functools.cache
def _invmap_padded_kernel(src_row:UOp, dest_row:UOp, counts:UOp, off:UOp, *, dname:str) -> UOp:
  G, M = src_row.shape
  G2, Mk = dest_row.shape
  E = counts.shape[1]
  assert G == G2 == counts.shape[0] == off.shape[0] and off.shape[1] == E + 1
  threads = UOp.special(256, "lidx0")
  workgroups = UOp.special(G * ((M + 255) // 256), "gidx0")
  sink = UOp.sink(src_row.base, dest_row.base, counts.base, off.base, threads, workgroups,
                  arg=KernelInfo(f"moe_invmap_padded_{M}_{Mk}_{E}", estimates=Estimates(ops=G*M*E, mem=G*(M+Mk)*4)))
  amd = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (amd/"moe_invmap_padded.cpp").read_text()
  lib = HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", f"-DG_DIM={G}", f"-DM_DIM={M}", f"-DMK_DIM={Mk}",
                                 f"-DE_DIM={E}", "-DTHREADS=256"]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def build_src_row_padded(dest_row:Tensor, counts:Tensor, off:Tensor, m_l:int) -> Tensor:
  buf = _sharded_invalids((dest_row.shape[0], m_l), dtypes.int32, dest_row.device)
  return Tensor.custom_kernel(buf, dest_row, counts, off,
    fxn=functools.partial(_invmap_padded_kernel, dname=str(dest_row.device)))[0]

@functools.cache
def _dispatch_gather_fwd_kernel(out:UOp, x:UOp, src_row:UOp, dest_row:UOp, k:int) -> UOp:
  # coalesced-write gather (replaces scatter+zero-init): grouped[m] = x[src_row[m]//k] (token), 0 for pad rows.
  # Writes EVERY grouped row so no separate zero-init kernel is needed (1 kernel vs the scatter path's 2).
  # dest_row is a saved input (referenced trivially below so the buffer-planner keeps it) for the module-level
  # backward to read from kernel.src -- avoids a grad_fxn closure over the live Routing (that cycles the assign graph).
  G, m_l, D = out.shape
  g, m, j, jo, ji = _kv_ranges(G, m_l, D, _blk_for(D))
  s = src_row.index(g, m).load().cast(dtypes.weakint)
  # keep dest_row in the sink without changing the output: OR its (invariant) row-0 index bit into the sentinel test.
  # dest_row values are >=0, so (dest_row[g,0] >= 0) is always True -> `valid` is unchanged; the read anchors the input.
  anchor = dest_row.index(g, UOp.const(0, dtypes.weakint)).load().cast(dtypes.weakint) >= 0
  valid = (s >= 0) & anchor
  t = valid.where(s, s.const_like(0)) // k                 # token index; clamp pad -> 0 (result zeroed below)
  val = valid.where(x.index(g, t, j).load(), UOp.const(0, x.dtype))
  return out.index(g, m, j).store(val).end(g, m, jo, ji).sink(
    arg=KernelInfo(name=f"dispatch_gather_{m_l}_{D}", opts_to_apply=GLUE_OPTS, beam=GLUE_BEAM))

def _dispatch_gather_bwd(gradient:UOp, kernel:UOp) -> tuple:
  # backward of the permute (a gather, not a scatter): d_x[t] = sum_j d_grouped[dest_row[t,j]] via grouped_gather_rows
  # (coalesced). Reads dest_row from kernel.src (saved fwd input) -- NO closure over the Routing (that cycled).
  _, x_u, _src_u, dest_u = kernel.src[1:5]
  dev = x_u.device
  G, T_l, D = x_u.shape
  k = dest_u.shape[1] // T_l
  sel = grouped_gather_rows(Tensor(gradient, device=dev), Tensor(dest_u, device=dev), G)
  return (None, sel.reshape(G, T_l, k, D).sum(2).cast(x_u.dtype).uop, None, None)

def m_max_for(t_local:int, experts_per_tok:int, n_experts:int) -> int:
  return (-(-t_local * experts_per_tok // BLOCK_ROW) + n_experts) * BLOCK_ROW

class Routing:
  def __init__(self, weights:Tensor, dest_row:Tensor, off:Tensor, m_l:int, n_groups:int, t_local:int,
               src_row:Tensor|None=None, topi:Tensor|None=None, counts:Tensor|None=None):
    self.weights, self.dest_row = weights, dest_row
    self.off, self.topi = off, topi
    self.counts = counts
    self.m_l, self.n_groups, self.t_local = m_l, n_groups, t_local
    self.src_row = src_row   # grouped row -> expanded source i; dispatch builds it only when routing did not supply one

  @property
  def tile_e(self) -> Tensor:
    # expert id per 256-row grouped tile: (G*m_l//BLOCK_ROW,). rows_e = this expanded x BLOCK_ROW.
    G, E = self.off.shape[0], self.off.shape[1] - 1
    tr = Tensor.arange(self.m_l // BLOCK_ROW, dtype=dtypes.int32).reshape(1, -1, 1) * BLOCK_ROW
    tr = tr.shard(self.off.device) if isinstance(self.off.device, tuple) else tr.to(self.off.device)
    return ((tr >= self.off[:, :E].reshape(G, 1, E)).sum(-1) - 1).cast(dtypes.int32).reshape(-1)

  @property
  def rows_e(self) -> Tensor:
    return self.tile_e.reshape(-1, 1).expand(-1, BLOCK_ROW).reshape(-1)

def n_groups_of(t:Tensor) -> int:
  return len(t.device) if isinstance(t.device, tuple) else 1

def route(logits:Tensor, experts_per_tok:int, n_experts:int, topk_out:tuple[Tensor, Tensor]|None=None) -> Routing:
  T, E = logits.shape
  k, G = experts_per_tok, n_groups_of(logits)
  assert T % G == 0, f"tokens {T} must split across {G} devices"
  T_l = T // G

  if getenv("FUSED_ROUTER_TOPK", 0):
    from extra.gptoss_kernels.router_topk import fused_router_topk
    assert k == 4 and E == 32
    weights, topi = fused_router_topk(logits.reshape(G, T_l, E),
      out=None if topk_out is None else tuple(t.reshape(G, T_l, k) for t in topk_out))
  else:
    topv, topi = logits.reshape(G, T_l, E).topk(k)
    weights = topv.softmax(-1)
  return route_topk(weights, topi, n_experts)

def route_topk(weights:Tensor, topi:Tensor, n_experts:int) -> Routing:
  G, T_l, k = weights.shape
  E, m_l = n_experts, m_max_for(T_l, k, n_experts)
  if (T_l, k, E) == (16384, 4, 32) and dname_of(topi.device) == "AMD":
    from extra.gptoss_kernels.routing import routing_maps
    dest_row, src_row, counts, off = routing_maps(topi)
    return Routing(weights, dest_row, off, m_l, G, T_l, src_row=src_row, topi=topi, counts=counts)
  m = topi.reshape(G, T_l * k).cast(dtypes.int32).one_hot(E).cast(dtypes.int32)

  counts = m.sum(1)
  pad = ((counts + (BLOCK_ROW - 1)) // BLOCK_ROW) * BLOCK_ROW
  off = pad.cumsum(1).pad(((0, 0), (1, 0)))
  dest_row = ((m.cumsum(1) + off[:, :E].reshape(G, 1, E)) * m).sum(-1).sub(1).cast(dtypes.int32)
  return Routing(weights, dest_row, off, m_l, G, T_l, topi=topi, counts=counts)

def dispatch(x:Tensor, r:Routing) -> Tensor:
  G, D = r.n_groups, x.shape[-1]
  if DISPATCH_GATHER: return dispatch_gather(x, r)
  return grouped_scatter_rows(x.reshape(G, r.t_local, D), r.dest_row, r.m_l).reshape(G * r.m_l, D)

def dispatch_gather(x:Tensor, r:Routing) -> Tensor:
  # NVIDIA-style permute: forward = coalesced-write gather via the inverse map (src_row); backward = gather via the
  # forward map (dest_row), d_x[t] = sum_j d_grouped[dest_row[t,j]] (identical to grouped_scatter_rows' backward).
  # Both directions are gathers (scatter-free), and the forward drops the zero-init kernel. Byte-exact with dispatch().
  # Reuse routing's inverse map when present; dest_row is a saved input for the module-level backward.
  G, D = r.n_groups, x.shape[-1]
  k = r.weights.shape[2]
  src_row = r.src_row if r.src_row is not None else build_src_row(r.dest_row, r.m_l)
  out = _sharded_invalids((G, r.m_l, D), x.dtype, x.device)
  out, *_ = Tensor.custom_kernel(out, x.reshape(G, r.t_local, D), src_row, r.dest_row,
                                 fxn=functools.partial(_dispatch_gather_fwd_kernel, k=k), grad_fxn=_dispatch_gather_bwd)
  return out.reshape(G * r.m_l, D)

@functools.cache
def _dispatch_fp8_dual_kernel(q_row:UOp, row_scale:UOp, q_col:UOp, si_col:UOp, q_in:UOp, e8_in:UOp,
                              src_row:UOp, _dest_row:UOp, *, k:int, dname:str) -> UOp:
  # Keep the row-major outputs physically flat. run_layer saves/consumes these as (G*M,D); giving the custom
  # output that exact shape lets the precompiled boundary retain the output buffer instead of copying the entire
  # fp8 activation merely to erase the singleton per-device G dimension.
  G, T, D = q_in.shape
  assert len(q_row.shape) == 2 and q_row.shape[0] % G == 0 and q_row.shape[1] == D
  M = q_row.shape[0] // G
  packed_row_si = row_scale.dtype == dtypes.uint32
  if packed_row_si: assert row_scale.shape == (D // 128, G * M)
  else: assert row_scale.shape == (G * M, D // 32)
  assert e8_in.shape == (G, T, D // 32)
  assert q_col.shape == (D, G * M) and si_col.shape == (G * M // 128, D)
  threads_per_wg = GPTOSS_DISPATCH_THREADS
  threads = UOp.special(threads_per_wg, "lidx0")
  split_m32 = bool(getenv("DISPATCH_DUAL_M32", 0))
  # The exact GPT-OSS M32/t128 producer appends coalesced row-SI-only workgroups to this launch. This keeps the
  # scale-major gather mapping while deleting the separate ~25 us launch; it also gives clang a slightly smaller
  # dispatch body (74 vs 76 VGPR). Keep non-production layouts on the standalone kernel.
  append_row_si = packed_row_si and bool(getenv("DISPATCH_APPEND_ROW_SI", 1)) and split_m32 and threads_per_wg == 128
  dispatch_wgs = G * (M // (32 if split_m32 else 128)) * (D // threads_per_wg)
  row_si_wgs = math.ceil((D // 128) * G * M / threads_per_wg) if append_row_si else 0
  workgroups = UOp.special(dispatch_wgs + row_si_wgs, "gidx0")
  elems = G * M * D
  sink = UOp.sink(q_row.base, row_scale.base, q_col.base, si_col.base, q_in.base, e8_in.base, src_row.base,
                  threads, workgroups,
                  arg=KernelInfo(f"dispatch_fp8_dual{'_m32' if split_m32 else ''}_{M}_{T}_{D}",
                                 estimates=Estimates(ops=2 * elems, mem=2 * elems + 3 * elems // 32)))
  amd = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (amd/"dispatch_fp8_dual.cpp").read_text()
  lib = HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", f"-DG_DIM={G}", f"-DM_DIM={M}", f"-DT_DIM={T}",
                                 f"-DD_DIM={D}", f"-DTOPK={k}", f"-DTHREADS={threads_per_wg}",
                                 f"-DNATIVE_MXFP8_CVT={getenv('NATIVE_MXFP8_CVT', 0)}",
                                 f"-DDISPATCH_ROW_SI={int(packed_row_si)}",
                                 f"-DDISPATCH_APPEND_ROW_SI={int(append_row_si)}",
                                 f"-DDISPATCH_DUAL_M32={int(split_m32)}"]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def _dispatch_row_si_kernel(row_si:UOp, e8_in:UOp, src_row:UOp, *, k:int, dname:str) -> UOp:
  G, T, SB = e8_in.shape
  assert SB % 4 == 0 and src_row.shape[0] == G and row_si.dtype == dtypes.uint32
  M = src_row.shape[1]
  assert row_si.shape == (SB // 4, G * M)
  elems = math.prod(row_si.shape)
  threads = UOp.special(256, "lidx0")
  workgroups = UOp.special(math.ceil(elems / 256), "gidx0")
  sink = UOp.sink(row_si.base, e8_in.base, src_row.base, threads, workgroups,
                  arg=KernelInfo(f"dispatch_row_si_{M}_{T}_{SB * 32}",
                                 estimates=Estimates(ops=elems, mem=elems * 12)))
  amd = pathlib.Path(__file__).parent.parent/"thunder"/"amd"
  src = (amd/"dispatch_row_si.cpp").read_text()
  lib = HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", f"-DG_DIM={G}", f"-DM_DIM={M}", f"-DT_DIM={T}",
                                 f"-DD_DIM={SB * 32}", f"-DTOPK={k}", "-DTHREADS=256"]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def _dispatch_fp8_dual_bwd(gradient:UOp, kernel:UOp) -> tuple:
  qrow_u, _erow_u, _qcol_u, _si_u, qin_u, _ein_u, _src_u, dest_u = kernel.src[1:9]
  dev = qin_u.device
  G, T, D = qin_u.shape
  M, k = qrow_u.shape[0] // G, dest_u.shape[1] // T
  grad_qin = grouped_gather_sum_rows(Tensor(gradient, device=dev).reshape(G, M, D), Tensor(dest_u, device=dev), G, k)
  return None, None, None, None, grad_qin.uop, None, None, None

def _dispatch_fp8_dual(xq:Tensor, xe8:Tensor, r:Routing, colw_out:tuple[Tensor, Tensor]|None=None) -> tuple[Tensor, ...]:
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE
  G, D, k = r.n_groups, xq.shape[-1], r.dest_row.shape[1] // r.t_local
  qin, ein = xq.reshape(G, r.t_local, D), xe8.reshape(G, r.t_local, D // 32)
  src_row = r.src_row
  if src_row is None:
    src_row = build_src_row_padded(r.dest_row, r.counts, r.off, r.m_l) if r.counts is not None else build_src_row(r.dest_row, r.m_l)
  # These are the shapes returned to FC1 and saved by run_layer. Allocating the custom outputs in those same
  # shapes avoids a full output materialization at the precompiled function boundary.
  qrow = owned_empty(_sharded_invalids((G * r.m_l, D), FP8_DTYPE, xq.device))
  row_si_req = getenv("FC1_DISPATCH_ROW_SI", 0)
  packed_row_si = bool(row_si_req) and ((r.m_l, r.t_local, D, k) == (73728, 16384, 3072, 4) or row_si_req == 2)
  append_row_si = packed_row_si and bool(getenv("DISPATCH_APPEND_ROW_SI", 1)) \
    and bool(getenv("DISPATCH_DUAL_M32", 0)) and GPTOSS_DISPATCH_THREADS == 128
  row_scale = alloc_like((D // 128, G * r.m_l), dtypes.uint32, xq.device, 1) if packed_row_si else \
    _sharded_invalids((G * r.m_l, D // 32), dtypes.uint8, xq.device)
  row_scale = owned_empty(row_scale)
  if colw_out is None:
    qcol = owned_empty(alloc_like((D, G * r.m_l), FP8_DTYPE, xq.device, 1))
    sicol = owned_empty(alloc_like((G * r.m_l // 128, D), dtypes.uint32, xq.device, 0))
  else:
    qcol, sicol = colw_out
    assert qcol.shape == (D, G * r.m_l) and qcol.dtype == FP8_DTYPE and qcol.device == xq.device
    assert sicol.shape == (G * r.m_l // 128, D) and sicol.dtype == dtypes.uint32 and sicol.device == xq.device
  qrow, row_scale, qcol, sicol, *_ = Tensor.custom_kernel(qrow, row_scale, qcol, sicol, qin, ein, src_row, r.dest_row,
    fxn=functools.partial(_dispatch_fp8_dual_kernel, k=k, dname=str(xq.device)), grad_fxn=_dispatch_fp8_dual_bwd)
  if packed_row_si and not append_row_si:
    row_scale = Tensor.custom_kernel(row_scale, ein, src_row,
      fxn=functools.partial(_dispatch_row_si_kernel, k=k, dname=str(xq.device)))[0]
  # Exposing the original per-token e8 as an @function output lets the precompiled backward substitute the saved
  # scale and delete its redundant per-layer scale materialization. FC1 consumes only the first four entries,
  # while run_layer saves every dispatch tuple entry.
  return (qrow, row_scale, qcol, sicol, xe8) if getenv("SAVE_DISPATCH_E8", 0) else (qrow, row_scale, qcol, sicol)

def dispatch_fp8(x:Tensor, r:Routing, colw_out:tuple[Tensor, Tensor]|None=None,
                 *, quantized:tuple[Tensor, Tensor]|None=None) -> tuple[Tensor, ...]:
  # FP8 dispatch: quantize the tokens to fp8 + e8m0 scales BEFORE the permute so the scatter moves fp8 (1 byte)
  # and the tiny e8 scales instead of bf16 (2 bytes), and the fc1 consumes the already-quantized grouped rows via
  # its (x_q, x_e8) tuple path (no re-quantize). Byte-identical operands to dispatch()->quantize_mxfp8(): the
  # scatter is a pure per-row permute and quantize_mxfp8 is per-row, so quantizing pre- or post-permute rounds the
  # same row identically; unwritten (pad) grouped rows are 0/e8=0 either way. Backward: fc1 returns grad_xq (bf16)
  # for the fp8 rows -> grouped_scatter_rows_hi gathers+sums it in bf16 -> quantize_mxfp8's STE backward scales by
  # qscale=2^(127-e8) (a per-token power of 2). That uniform pow2 factor commutes with the sum-over-k, so the grad
  # reaching x is byte-exact with the current bf16-dispatch backward (which sums the already-physical grad).
  # This dispatch-only producer needs q and e8 from the same per-row block reduction.  The generic expression
  # materializes them as two full passes; the custom producer writes both with the exact physical shapes consumed
  # below (and preserves quantize_mxfp8's STE in its grad_fxn).
  if getenv("FUSED_DISPATCH_QE8", 1):
    from extra.llama_kernels.quantize_mxfp8_fused import quantize_mxfp8_fused_qe8 as quantize_mxfp8
  else:
    from extra.gemm.cdna_asm_gemm import quantize_mxfp8
  G, D = r.n_groups, x.shape[-1]
  xq, xe8 = quantize_mxfp8(x)[:2] if quantized is None else quantized
  if FUSED_DISPATCH_COLW: return _dispatch_fp8_dual(xq, xe8, r, colw_out)
  xg_q = grouped_scatter_rows_hi(xq.reshape(G, r.t_local, D), r.dest_row, r.m_l).reshape(G * r.m_l, D)
  xg_e8 = grouped_scatter_rows(xe8.reshape(G, r.t_local, D // 32), r.dest_row, r.m_l).reshape(G * r.m_l, D // 32)
  return (xg_q, xg_e8)

def combine(y:Tensor, r:Routing, n_tokens:int, experts_per_tok:int) -> Tensor:
  G, D, k = r.n_groups, y.shape[-1], experts_per_tok
  sel = grouped_gather_rows(y.reshape(G, r.m_l, D), r.dest_row, G).reshape(G, r.t_local, k, D)
  return (sel * r.weights.reshape(G, r.t_local, k, 1).cast(sel.dtype)).sum(2).reshape(n_tokens, D).cast(y.dtype)
