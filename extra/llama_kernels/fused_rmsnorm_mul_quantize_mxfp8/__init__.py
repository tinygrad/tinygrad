from __future__ import annotations
import functools, math, os, pathlib, warnings
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.hipcc import HIPCCCompiler
from extra.gemm.cdna_asm_gemm import FP8_DTYPE
from extra.llama_kernels import NUM_WG, THREADS_PER_WG, alloc_like, alloc_local, compile_hip, dname_of

_SRC_DIR = pathlib.Path(__file__).parent
GPTOSS_COOP_EPILOGUE = os.getenv("GPTOSS_RMSNORM_MX_EP8", "0") == "1"

@functools.cache
def _custom_fwd(q:UOp, e8:UOp, rrms:UOp, x:UOp, weight:UOp, *, dname:str, eps:float) -> UOp:
  *lead, hidden = x.shape
  rows, padded = math.prod(lead), q.shape[-1]
  coop_epilogue = GPTOSS_COOP_EPILOGUE and (rows, hidden, padded) == (16384, 2880, 3072)
  num_wg = min(NUM_WG, rows)
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(num_wg, "gidx0")
  sink = UOp.sink(q.base, e8.base, rrms.base, x.base, weight.base, threads, workgroups,
                  arg=KernelInfo(f"rmsnorm_mul_quantize_mxfp8_{rows}_{hidden}_{padded}_ep8{int(coop_epilogue)}",
                                 estimates=Estimates(ops=8*rows*hidden,
                                                     mem=rows*(hidden*2+padded+padded//32+4)+hidden*2)))
  src = (_SRC_DIR/"rmsnorm_mul_quantize_mxfp8.cpp").read_text()
  defines = [f"-DN_ELEMS={rows*hidden}", f"-DHIDDEN={hidden}", f"-DPADDED={padded}", f"-DNUM_WG={num_wg}",
             f"-DTHREADS_PER_WG={THREADS_PER_WG}", f"-DEPS_LITERAL={eps}f", f"-DGPTOSS_COOP_EPILOGUE={int(coop_epilogue)}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_bwd(grad_x:UOp, grad_weight_partial:UOp, grad_q:UOp, x:UOp, weight:UOp, e8:UOp, rrms:UOp,
                *, dname:str) -> UOp:
  *lead, hidden = x.shape
  rows, padded = math.prod(lead), grad_q.shape[-1]
  gptoss_shape = (rows, hidden, padded) == (16384, 2880, 3072)
  rocm72_requested = gptoss_shape and os.getenv("GPTOSS_RMSNORM_MX_BWD_ROCM72", "0") == "1"
  rocm72_root = pathlib.Path("/opt/rocm-7.2.1")
  rocm72_hipcc = rocm72_root/"bin"/"hipcc"
  rocm72 = rocm72_requested and rocm72_hipcc.is_file() and (rocm72_root/"include"/"hip").is_dir()
  if rocm72_requested and not rocm72:
    warnings.warn("GPTOSS_RMSNORM_MX_BWD_ROCM72=1 but /opt/rocm-7.2.1 is unavailable; using the configured HIPCC toolchain",
                  RuntimeWarning)
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
  src = (_SRC_DIR/"rmsnorm_mul_quantize_mxfp8_bwd.cpp").read_text()
  defines = [f"-DN_ELEMS={rows*hidden}", f"-DHIDDEN={hidden}", f"-DPADDED={padded}", f"-DNUM_WG={num_wg}",
             f"-DTHREADS_PER_WG={THREADS_PER_WG}"]
  # ROCm 7.2.1 compiles the unchanged exact backward source to 90 instead of 170 VGPR.  Full poisoned/adversarial
  # output bits match 7.1; GPU1/7 ABBA64 improves 193.58/193.68 -> 114.51/114.79 us with no spills.
  lib = HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", *defines],
                      hipcc_path=rocm72_hipcc if rocm72 else None,
                      rocm_path=rocm72_root if rocm72 else None).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=lib)))

def _backward(gradient:UOp, kernel:UOp) -> tuple:
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
    fxn=functools.partial(_custom_bwd, dname=dname_of(device)))
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
  q, e8, rrms, *_ = Tensor.custom_kernel(q, e8, rrms, x, weight,
                                          fxn=functools.partial(_custom_fwd, dname=dname_of(x.device), eps=eps),
                                          grad_fxn=_backward)
  return q, e8, rrms
