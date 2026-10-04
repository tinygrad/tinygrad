from __future__ import annotations
import functools, math, os, pathlib
from extra.hipcc import HIPCCCompiler, rocm_toolchain, hip_program
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.gemm.cdna_asm_gemm import FP8_DTYPE
from extra.llama_kernels import NUM_WG, THREADS_PER_WG, alloc_like, alloc_local, compile_hip, dname_of, owned_empty

COOP_EPILOGUE = getenv("RMSNORM_MX_EP8")

def rmsnorm_mul_fwd(x_in:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  x = x_in.float()
  rrms = (x.square().mean(-1, keepdim=True) + eps).rsqrt()
  return ((x * rrms) * weight.float()).cast(x_in.dtype), rrms

@functools.cache
def _rmsnorm_mul_fwd_fxn(x_in_p, w_p, eps, device):
  return rmsnorm_mul_fwd(Tensor(x_in_p, device=device), Tensor(w_p, device=device), eps)

def _rmsnorm_mul_bwd(grad:UOp, call:UOp) -> tuple:
  return _rmsnorm_mul_bwd_inputs(grad, call.src[1], call.src[2], call.unbound_outputs[1])

@functools.cache
def _custom_rmsnorm_mul_bwd(grad_x:UOp, grad_weight_partial:UOp, grad:UOp, x:UOp, weight:UOp, rrms:UOp, *, dname:str) -> UOp:
  rows, hidden = math.prod(x.shape[:-1]), x.shape[-1]
  n_partials = grad_weight_partial.shape[0]
  # Exact GPTOSS: pair two of the 16 rows owned by each workgroup, halving barriers. Full grad-x/weight-partial bits match;
  # GPU1/7 JITBEAM3 ABBA64: 112.84/111.79us -> 98.82/97.98us (-12.43/-12.35%).
  pair_rows = getenv("GPTOSS_RMSNORM_MUL_BWD_PAIR", 0) and (rows, hidden, n_partials, THREADS_PER_WG) == (16384, 2880, 1024, 256)
  assert grad_x.shape == x.shape and grad.shape in (x.shape, (rows, hidden))
  assert grad.dtype == x.dtype == grad_x.dtype == dtypes.bfloat16
  assert weight.shape == (hidden,) and weight.dtype == dtypes.bfloat16 and rrms.shape == (*x.shape[:-1], 1)
  assert rrms.dtype == grad_weight_partial.dtype == dtypes.float32 and grad_weight_partial.shape == (n_partials, hidden)
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(n_partials, "gidx0")
  zero = UOp.const(0, dtypes.int32)
  accesses = (grad_x.index(zero).store(UOp.const(0, grad_x.dtype)),
              grad_weight_partial.index(zero).store(UOp.const(0.0, dtypes.float32)),
              grad.index(zero).load(), x.index(zero).load(), weight.index(zero).load(), rrms.index(zero).load())
  sink = UOp.sink(grad_x.base, grad_weight_partial.base, grad.base, x.base, weight.base, rrms.base,
                  *accesses, threads, workgroups,
                  arg=KernelInfo(f"rmsnorm_mul_bwd_pair{int(pair_rows)}_{rows}_{hidden}_{n_partials}",
                                 estimates=Estimates(ops=10*rows*hidden,
                                                     mem=rows*hidden*6+rows*4+n_partials*hidden*4+hidden*2)))
  src = (pathlib.Path(__file__).parent/"rmsnorm_mul_bwd.cpp").read_text()
  defines = [f"-DROWS={rows}", f"-DHIDDEN={hidden}", f"-DNUM_WG={n_partials}", f"-DTHREADS={THREADS_PER_WG}",
             f"-DGPTOSS_PAIR_ROWS={int(pair_rows)}"]
  return hip_program(sink, src, compile_hip(src, defines))

def rmsnorm_mul(x_in:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  fxn = _rmsnorm_mul_fwd_fxn(x_in.as_param(0).uop, weight.as_param(1).uop, eps, x_in.device)
  outputs = UOp.call_with_outputs((fxn[0].uop, fxn[1].uop), x_in.uop, weight.uop, grad_fxn=_rmsnorm_mul_bwd)
  return Tensor(outputs[0]), Tensor(outputs[1])

@functools.cache
def _custom_rmsnorm_mul_quantize_mxfp8_fwd(q:UOp, e8:UOp, rrms:UOp, x:UOp, weight:UOp, *, dname:str, eps:float) -> UOp:
  *lead, hidden = x.shape
  rows, padded = math.prod(lead), q.shape[-1]
  coop_epilogue = COOP_EPILOGUE and (rows, hidden, padded) == (16384, 2880, 3072)
  num_wg = min(NUM_WG, rows)
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(num_wg, "gidx0")
  sink = UOp.sink(q.base, e8.base, rrms.base, x.base, weight.base, threads, workgroups,
                  arg=KernelInfo(f"rmsnorm_mul_quantize_mxfp8_{rows}_{hidden}_{padded}_ep8{int(coop_epilogue)}",
                                 estimates=Estimates(ops=8*rows*hidden,
                                                     mem=rows*(hidden*2+padded+padded//32+4)+hidden*2)))
  src = (pathlib.Path(__file__).parent/"rmsnorm_mul_quantize_mxfp8.cpp").read_text()
  defines = [f"-DN_ELEMS={rows*hidden}", f"-DHIDDEN={hidden}", f"-DPADDED={padded}", f"-DNUM_WG={num_wg}",
             f"-DTHREADS_PER_WG={THREADS_PER_WG}", f"-DEPS_LITERAL={eps}f", f"-DCOOP_EPILOGUE={int(coop_epilogue)}"]
  return hip_program(sink, src, compile_hip(src, defines))

@functools.cache
def _custom_rmsnorm_mul_quantize_mxfp8_bwd(grad_x:UOp, grad_weight_partial:UOp, grad_q:UOp, x:UOp, weight:UOp, e8:UOp, rrms:UOp,
                *, dname:str) -> UOp:
  *lead, hidden = x.shape
  rows, padded = math.prod(lead), grad_q.shape[-1]
  gptoss_shape = (rows, hidden, padded) == (16384, 2880, 3072)
  rocm72_requested = gptoss_shape and os.getenv("GPTOSS_RMSNORM_MX_BWD_ROCM72", "0") == "1"
  rocm72_hipcc, rocm72_root = rocm_toolchain("/opt/rocm-7.2.1", rocm72_requested,
    flag="GPTOSS_RMSNORM_MX_BWD_ROCM72")
  rocm72 = rocm72_root is not None
  num_wg = min(NUM_WG, rows)
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(num_wg, "gidx0")
  zero = UOp.const(0, dtypes.int32)
  accesses = (grad_x.index(zero).store(UOp.const(0, grad_x.dtype)),
              grad_weight_partial.index(zero).store(UOp.const(0.0, dtypes.float32)),
              grad_q.index(zero).load(), x.index(zero).load(), weight.index(zero).load(),
              e8.index(zero).load(), rrms.index(zero).load())
  sink = UOp.sink(grad_x.base, grad_weight_partial.base, grad_q.base, x.base, weight.base, e8.base, rrms.base,
                  *accesses, threads, workgroups,
                  arg=KernelInfo(f"rmsnorm_mul_quantize_mxfp8_bwd_{rows}_{hidden}_{padded}_r72{int(rocm72)}",
                                 estimates=Estimates(ops=10*rows*hidden,
                                                     mem=rows*(hidden*6+padded*2+padded//32+4)+num_wg*hidden*4)))
  src = (pathlib.Path(__file__).parent/"rmsnorm_mul_quantize_mxfp8_bwd.cpp").read_text()
  defines = [f"-DN_ELEMS={rows*hidden}", f"-DHIDDEN={hidden}", f"-DPADDED={padded}", f"-DNUM_WG={num_wg}",
             f"-DTHREADS_PER_WG={THREADS_PER_WG}"]
  # ROCm 7.2.1 compiles the unchanged exact backward source to 90 instead of 170 VGPR.  Full poisoned/adversarial
  # output bits match 7.1; GPU1/7 ABBA64 improves 193.58/193.68 -> 114.51/114.79 us with no spills.
  lib = HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", *defines],
                      hipcc_path=rocm72_hipcc,
                      rocm_path=rocm72_root).compile_cached(src)
  return hip_program(sink, src, lib)

def _rmsnorm_mul_quantize_mxfp8_backward(gradient:UOp, kernel:UOp) -> tuple:
  _, e8_u, rrms_u, x_u, weight_u = kernel.src[1:]
  device = x_u.device
  axis = x_u.axis if isinstance(device, tuple) else None
  *lead, hidden = x_u.shape
  num_wg = min(NUM_WG, math.prod(lead))
  grad_x = alloc_like(x_u.shape, x_u.dtype, device, axis)
  grad_weight_partial = alloc_local((num_wg, hidden), dtypes.float32, device, axis)
  grad_q = Tensor(gradient, device=device).cast(dtypes.bfloat16).contiguous()
  grad_x, grad_weight_partial, *_ = Tensor.custom_kernel(
    grad_x, grad_weight_partial, grad_q, Tensor(x_u, device=device), Tensor(weight_u, device=device),
    Tensor(e8_u.after(kernel), device=device), Tensor(rrms_u.after(kernel), device=device),
    fxn=functools.partial(_custom_rmsnorm_mul_quantize_mxfp8_bwd, dname=dname_of(device)))
  grad_weight = grad_weight_partial.sum(0).cast(weight_u.dtype)
  return None, None, None, grad_x.uop, grad_weight.uop

def rmsnorm_mul_quantize_mxfp8(x:Tensor, weight:Tensor, eps:float, padded:int|None=None) -> tuple[Tensor, Tensor, Tensor]:
  """RMSNorm(x)*weight directly to rowwise MXFP8. Returns (q, e8, rrms), without a BF16 normalized round-trip."""
  assert x.dtype == weight.dtype == dtypes.bfloat16 and x.shape[-1] == weight.shape[0], f"{x.shape=} {weight.shape=}"
  hidden = x.shape[-1]
  padded = math.ceil(hidden / 256) * 256 if padded is None else padded
  assert padded >= hidden and padded % 256 == 0 and hidden % 32 == 0
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  q = alloc_like((*x.shape[:-1], padded), FP8_DTYPE, x.device, axis)
  e8 = alloc_like((*x.shape[:-1], padded // 32), dtypes.uint8, x.device, axis)
  rrms = alloc_like((*x.shape[:-1], 1), dtypes.float32, x.device, axis)
  # Own fresh destinations before the native write so precompiled returns reuse the producer storage.
  q, e8, rrms = (v.clone() for v in (q, e8, rrms))
  q, e8, rrms, *_ = Tensor.custom_kernel(q, e8, rrms, x, weight,
                                          fxn=functools.partial(_custom_rmsnorm_mul_quantize_mxfp8_fwd, dname=dname_of(x.device), eps=eps),
                                          grad_fxn=_rmsnorm_mul_quantize_mxfp8_backward)
  return q, e8, rrms

@functools.cache
def _custom_fast_final_denom(denom:UOp, rrms:UOp, x:UOp, *, dname:str, eps:float) -> UOp:
  local_x_shape = x.shard_shape
  rows, hidden = math.prod(local_x_shape[:-1]), local_x_shape[-1]
  threads = 64
  sink = UOp.sink(denom.base, rrms.base, x.base,
                  UOp.special(threads, "lidx0"), UOp.special(rows // threads, "gidx0"),
                  arg=KernelInfo(f"fast_final_rmsnorm_denom_rrms_{rows}_{hidden}",
                                 estimates=Estimates(ops=2*rows*hidden, mem=2*rows*hidden+8*rows)))
  src = (pathlib.Path(__file__).parent/"fast_final_denom.cpp").read_text()
  defines = [f"-DROWS={rows}", f"-DHIDDEN={hidden}", f"-DTHREADS={threads}", f"-DEPS_LITERAL={eps}f", "-fno-finite-math-only"]
  return hip_program(sink, src, compile_hip(src, defines))

def _fast_final_rmsnorm_fwd(x:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor, Tensor]:
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  local_shape = x.uop.shard_shape if axis is not None else x.shape
  assert axis in (None, 0) and math.prod(local_shape[:-1]) == 16384 and local_shape[-1] == 2880, \
    f"unsupported GPT-OSS final RMS tensor ABI {x.shape}/{x.uop.shard_shape} axis={axis}"
  denom = alloc_like(x.shape[:-1], dtypes.float32, x.device, axis).clone()
  rrms = alloc_like(x.shape[:-1], dtypes.float32, x.device, axis).clone()
  denom, rrms, *_ = Tensor.custom_kernel(denom, rrms, x, fxn=functools.partial(
    _custom_fast_final_denom, dname=dname_of(x.device), eps=eps))
  x_normed = (x.float() / denom.unsqueeze(-1)).cast(x.dtype)
  return x_normed * weight, denom, rrms

@functools.cache
def _fast_final_rmsnorm_fwd_fxn(x_p, weight_p, eps, device):
  return _fast_final_rmsnorm_fwd(Tensor(x_p, device=device), Tensor(weight_p, device=device), eps)

def _fast_final_rmsnorm_bwd(gradient:UOp, call:UOp, *, eps:float) -> tuple:
  x, weight = Tensor(call.src[1]), Tensor(call.src[2])
  denom = Tensor(call.unbound_outputs[1])
  rrms = Tensor(call.unbound_outputs[2])
  grad = Tensor(gradient, device=x.device)
  xf = x.float()
  rrms_ref = (xf.square().mean(-1, keepdim=True) + eps).rsqrt()
  y_ref = (xf * rrms_ref).cast(x.dtype) * weight
  d_x, d_weight = y_ref.gradient(x, weight, gradient=grad)
  denom_saved = denom.unsqueeze(-1)
  rrms_from_denom = 1.0 / denom_saved
  rrms_saved = rrms.unsqueeze(-1)
  replace_dx = {rrms_ref.uop:rrms_from_denom.uop, rrms_ref.uop.src[0]:denom_saved.uop}
  replace_dw = {rrms_ref.uop:rrms_saved.uop}
  return d_x.uop.substitute(replace_dx, walk=True), d_weight.uop.substitute(replace_dw, walk=True)

def fast_final_rmsnorm(x:Tensor, weight:Tensor, eps:float) -> Tensor:
  """GPT-OSS final RMSNorm, with a specialized denominator pass for the training shape."""
  assert x.dtype == weight.dtype == dtypes.bfloat16 and x.shape[-1] == weight.shape[0] == 2880
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  local_shape = x.uop.shard_shape if axis is not None else x.shape
  if axis not in (None, 0) or math.prod(local_shape[:-1]) != 16384:
    xf = x.float()
    return (xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt()).cast(x.dtype) * weight
  fxn = _fast_final_rmsnorm_fwd_fxn(x.as_param(0).uop, weight.as_param(1).uop, eps, x.device)
  outputs = UOp.call_with_outputs((fxn[0].uop, fxn[1].uop, fxn[2].uop),
    x.uop, weight.uop, grad_fxn=functools.partial(_fast_final_rmsnorm_bwd, eps=eps))
  return Tensor(outputs[0])


@functools.cache
def _custom_gptoss_residual_mul_fwd(out:UOp, rrms:UOp, saved_h:UOp, x:UOp, proj:UOp, bias:UOp, weight:UOp, *, eps:float) -> UOp:
  assert out.shard_shape == saved_h.shard_shape == x.shard_shape and math.prod(x.shard_shape[:-1]) == 16384
  assert x.shard_shape[-1] == 2880 and proj.shard_shape == (16384,3072) and bias.shape == weight.shape == (2880,)
  assert all(a.dtype == dtypes.bfloat16 for a in (out,saved_h,x,proj,bias,weight)) and rrms.dtype == dtypes.float32
  zero = UOp.const(0,dtypes.int32)
  accesses = (*(a.index(zero).store(UOp.const(0,a.dtype)) for a in (out,rrms,saved_h)),
              *(a.index(zero).load() for a in (x,proj,bias,weight)))
  shuffle = bool(getenv('GPTOSS_RESIDUAL_RMS_SHUFFLE', 0))
  sink = UOp.sink(*(a.base for a in (out,rrms,saved_h,x,proj,bias,weight)),*accesses,
                  UOp.special(64,'lidx0'),UOp.special(8192,'gidx0'),
                  arg=KernelInfo('gptoss_residual_rmsnorm_mul_fwd'+('_shuffle' if shuffle else ''),
                    estimates=Estimates(ops=7*16384*2880,mem=8*16384*2880+4*16384+4*2880)))
  source = (pathlib.Path(__file__).parent/'gptoss_residual_mul_fwd.cpp').read_text()
  lib = compile_hip(source,[f'-DEPS_LITERAL={eps}f','-fno-fast-math','-ffp-contract=off']+
                    (['-DGPTOSS_RESIDUAL_RMS_SHUFFLE=1'] if shuffle else []))
  return hip_program(sink, source, lib)


def _gptoss_residual_mul_fwd_bwd(grad:UOp, call:UOp) -> tuple:
  from extra.llama_kernels.dense_bias import dense_bias_backward
  _,rrms,h,x,proj,bias,weight = call.src[1:]
  dh,dweight = _rmsnorm_mul_bwd_inputs(grad,h.after(call),weight,rrms.after(call))
  # Preserve the existing dense-bias backward and its transpose-quantization mailbox for the WO GEMM.
  proj_view = Tensor(proj)[:,:2880].reshape(x.shape)
  dproj,dbias = dense_bias_backward(dh,proj_view.uop,bias)
  dproj = Tensor(dproj).reshape(-1,2880).pad(((0,0),(0,192)))
  return None,None,None,dh,dproj.uop,dbias,dweight


def gptoss_residual_rmsnorm_mul(h:Tensor, weight:Tensor, eps:float) -> tuple[Tensor,Tensor,Tensor]|None:
  """Normalize the exact attention residual and save it in the same pass. Only y is differentiated."""
  if math.prod(h.uop.shard_shape[:-1]) != 16384 or (inputs := _gptoss_residual_inputs(h.uop)) is None: return None
  x,proj,bias = inputs
  axis = x.uop.axis if isinstance(x.device,tuple) else None
  out = owned_empty(alloc_like(x.shape,dtypes.bfloat16,x.device,axis))
  rrms = owned_empty(alloc_like((*x.shape[:-1],1),dtypes.float32,x.device,axis))
  saved_h = owned_empty(alloc_like(x.shape,dtypes.bfloat16,x.device,axis))
  out,rrms,saved_h,*_ = Tensor.custom_kernel(out,rrms,saved_h,x,proj,bias,weight,
    fxn=functools.partial(_custom_gptoss_residual_mul_fwd,eps=eps),grad_fxn=_gptoss_residual_mul_fwd_bwd)
  return out,rrms,saved_h


def _rmsnorm_mul_bwd_inputs(grad:UOp, x_u:UOp, weight_u:UOp, rrms_u:UOp) -> tuple:
  if getenv("FUSED_RMSNORM_MUL_BWD", 0):
    x, weight, rrms = Tensor(x_u), Tensor(weight_u), Tensor(rrms_u)
    g = Tensor(grad, device=x_u.device)
    assert x.dtype == weight.dtype == g.dtype == dtypes.bfloat16 and weight.shape == (x.shape[-1],)
    device, axis = x.device, (x.uop.axis if isinstance(x.device, tuple) else None)
    local_rows = math.prod(x.uop.shard_shape[:-1] if axis is not None else x.shape[:-1])
    n_partials = min(NUM_WG, local_rows)
    grad_x = owned_empty(alloc_like(x.shape, dtypes.bfloat16, device, axis))
    partial_shape = (n_partials * len(device), x.shape[-1]) if isinstance(device, tuple) and axis is not None else (n_partials, x.shape[-1])
    grad_weight_partial = alloc_like(partial_shape, dtypes.float32, device, 0 if isinstance(device, tuple) and axis is not None else None)
    grad_weight_partial = owned_empty(grad_weight_partial)
    # The router emits a physical [tokens, hidden] gradient. Making its logical 3D view contiguous would copy it.
    flat_grad = getenv("GPTOSS_RMSNORM_BWD_FLAT_GRAD", 0) and (local_rows, x.shape[-1]) == (16384, 2880)
    kernel_grad = g.reshape(-1, x.shape[-1]) if flat_grad else g.contiguous()
    grad_x, grad_weight_partial, *_ = Tensor.custom_kernel(
      grad_x, grad_weight_partial, kernel_grad, x, weight, rrms,
      fxn=functools.partial(_custom_rmsnorm_mul_bwd, dname=dname_of(device)))
    return grad_x.uop, grad_weight_partial.sum(0).cast(weight_u.dtype).uop
  x, weight, rrms = Tensor(x_u).float(), Tensor(weight_u).float(), Tensor(rrms_u)
  x_normed = x * rrms                                  # recompute unweighted normed
  d_y = Tensor(grad).float()
  dxn = d_y * weight                                   # d/d(x_normed)
  d_x = rrms * (dxn - x_normed * (dxn * x_normed).mean(-1, keepdim=True))
  dw = d_y * x_normed
  d_weight = dw.sum(axis=tuple(range(dw.ndim - 1)))    # reduce batch/seq -> [dim] (allreduces the DP batch axis)
  return (d_x.cast(x_u.dtype).uop, d_weight.cast(weight_u.dtype).uop)


def _const_tuple(u:UOp) -> tuple[int, ...]|None:
  if u.op is not Ops.STACK or any(x.op is not Ops.CONST or not isinstance(x.arg, int) for x in u.src): return None
  return tuple(x.arg for x in u.src)


def _gptoss_residual_inputs(x_u:UOp) -> tuple[Tensor, Tensor, Tensor]|None:
  # Match only bf16(residual + dense_bias(padded_projection[:,:2880], bias)). The logical projection
  # remains in x_u's autograd graph; the HIP callback receives its physical 3072-wide parent instead.
  if x_u.op is not Ops.ADD or x_u.dtype != dtypes.bfloat16 or x_u.shape[-1] != 2880 or len(x_u.src) != 2: return None
  from extra.llama_kernels.dense_bias import _dense_bias_bwd
  for i, candidate in enumerate(x_u.src):
    if candidate.op is not Ops.AFTER or candidate.src[1].op is not Ops.CALL: continue
    dense_call = candidate.src[1]
    if len(dense_call.src) != 4 or dense_call.arg.grad_fxn is not _dense_bias_bwd: continue
    if candidate is not dense_call.unbound_outputs[0]: continue
    body = dense_call.src[0]
    if body.op is not Ops.SINK or len(body.src) != 1 or body.src[0].op is not Ops.STORE: continue
    value = body.src[0].src[1]
    if value.op is not Ops.ADD or {u.arg.slot for u in value.toposort(enter_calls=False) if u.op is Ops.PARAM} != {0, 1}: continue
    residual, proj, bias = Tensor(x_u.src[1-i]), Tensor(dense_call.src[1]), Tensor(dense_call.src[2])
    if residual.shape != x_u.shape or proj.shape != x_u.shape or bias.shape != (2880,): continue
    if residual.dtype != dtypes.bfloat16 or proj.dtype != dtypes.bfloat16 or bias.dtype != dtypes.bfloat16: continue

    # matmul_mx returns reshape(shrink([rows,3072], [0:rows,0:2880])). Reject any other view instead
    # of guessing a pitch from max_numel: that was unsafe when a generic SHRINK was compacted.
    rows = math.prod(x_u.shape[:-1])
    proj_2d = proj.uop.src[0] if proj.uop.op is Ops.RESHAPE and proj.uop.src[0].shape == (rows, 2880) else proj.uop
    if proj_2d.op is not Ops.SHRINK or proj_2d.shape != (rows, 2880): continue
    if _const_tuple(proj_2d.src[1]) != (0, 0) or _const_tuple(proj_2d.src[2]) != (rows, 2880): continue
    proj_phys_u = proj_2d.src[0]
    if proj_phys_u.shape != (rows, 3072) or proj_phys_u.dtype != dtypes.bfloat16: continue
    proj_phys = Tensor(proj_phys_u)
    if proj_phys.device != residual.device or bias.device != residual.device: continue
    return residual, proj_phys, bias
  return None
