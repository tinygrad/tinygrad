from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import FP8_MAX, NUM_WG, THREADS_PER_WG, compile_cpp, shard_shape, scalar_amax

@functools.cache
def _custom_mul_quantize_fp8(fp8_out:UOp, amax_buf:UOp, x:UOp, weight:UOp, amax_state:UOp, dname:str) -> UOp:
  MBS, SEQ, HIDDEN = x.shape
  n_elems = MBS * SEQ * HIDDEN
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(NUM_WG, "gidx0")
  mem = n_elems * 2 + HIDDEN * 2 + n_elems + NUM_WG * 2
  sink = UOp.sink(fp8_out.base, amax_buf.base, x.base, weight.base, amax_state.base, threads, workgroups,
                  arg=KernelInfo(f"fused_mul_quantize_fp8_{n_elems}_h{HIDDEN}", estimates=Estimates(ops=3*n_elems, mem=mem)))
  src, lib = compile_cpp(pathlib.Path(__file__).parent, "fused_mul_quantize_fp8.cpp", n_elems, HIDDEN)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def _fused_mul_quantize_fp8_bwd(gradient:UOp, kernel:UOp):
  # NOTE: inputs are (fp8_out, amax_buf, x, weight, amax_state); grads for x and weight
  _, _, x_u, weight_u, amax_state_u = kernel.src[1:]
  device = x_u.device
  grad_t = Tensor(gradient, device=device).cast(dtypes.bfloat16)
  x_t, weight_t = Tensor(x_u, device=device), Tensor(weight_u, device=device)
  scale = FP8_MAX / (Tensor(amax_state_u, device=device).float() + 1e-8)
  grad_scaled = grad_t.float() * scale
  # NOTE: grad_x stays bf16 to avoid CSE materializing a (MBS, SEQ, HIDDEN) fp32 intermediate
  grad_x = (grad_scaled * weight_t.float()).cast(dtypes.bfloat16)
  grad_weight = (grad_scaled * x_t.float()).sum(axis=(0, 1)).cast(dtypes.bfloat16)
  return (None, None, grad_x.uop, grad_weight.uop, None)

def fused_mul_quantize_fp8(x:Tensor, weight:Tensor, amax_state:Tensor, fp8_dtype) -> tuple[Tensor, Tensor, Tensor]:
  # NOTE: (x * weight) -> fp8 + amax, delayed scaling. Returns (fp8, inv_scale, new_amax)
  assert x.dtype == dtypes.bfloat16 and weight.dtype == dtypes.bfloat16
  assert x.shape[-1] == weight.shape[-1], f"HIDDEN mismatch: x={x.shape}, weight={weight.shape}"
  MBS, SEQ, HIDDEN = x.shape
  if isinstance(x.device, tuple):
    axis, ndev = x.uop.axis, len(x.device)
    assert axis in (0, 1), f"unsupported sharding axis={axis}"
    fp8_out = Tensor(Tensor.invalids(*shard_shape((MBS, SEQ, HIDDEN), axis, ndev), dtype=fp8_dtype,
                                     device=x.device).uop.multi(axis), device=x.device)
    amax_buf = Tensor(Tensor.invalids(NUM_WG, dtype=dtypes.bfloat16, device=x.device).uop.multi(0), device=x.device)
    dname = x.device[0].split(":")[0]
  else:
    fp8_out = Tensor.invalids(MBS, SEQ, HIDDEN, dtype=fp8_dtype, device=x.device)
    amax_buf = Tensor.invalids(NUM_WG, dtype=dtypes.bfloat16, device=x.device)
    dname = x.device.split(":")[0] if isinstance(x.device, str) else x.device
  fxn = functools.partial(_custom_mul_quantize_fp8, dname=dname)
  fp8_out, amax_buf, *_ = Tensor.custom_kernel(fp8_out, amax_buf, x, weight, amax_state, fxn=fxn,
                                                grad_fxn=_fused_mul_quantize_fp8_bwd)
  new_amax = scalar_amax(amax_buf)
  inv_scale = (amax_state.float() + 1e-8) / FP8_MAX
  return fp8_out, inv_scale, new_amax
