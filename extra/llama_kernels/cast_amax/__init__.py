from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import NUM_WG, THREADS_PER_WG, compile_cpp, alloc_like, zero_scalar, dname_of

# module-level mailbox: grad_xw13 UOp -> (grad_xw13_fp8 UOp, delayed amax UOp)
# lets cdna_asm_gemm's bwd reuse the fp8 companion produced by the fused silu_mul bwd kernel
# instead of doing a redundant bf16 -> fp8 quantize.
_grad_fp8_mailbox:dict[UOp, tuple[UOp, UOp]] = {}

@functools.cache
def _custom_fused_bwd_w13(grad_xw13_fp8:UOp, grad_amax_next:UOp,
                          xw13:UOp, grad_x2:UOp, amax_state:UOp, grad_amax_state:UOp, layer_num:UOp|None=None, dname:str=None) -> UOp:
  hidden = xw13.shape[2] // 2
  n_elems = xw13.shape[0] * xw13.shape[1] * hidden
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(NUM_WG, "gidx0")
  mem = n_elems * 2 * 3 + n_elems * 2 + 4
  sink = UOp.sink(grad_xw13_fp8.base, grad_amax_next.base,
                  xw13.base, grad_x2.base, amax_state.base, grad_amax_state.base, *((layer_num.base,) if layer_num is not None else ()), threads, workgroups,
                  arg=KernelInfo(f"fused_silu_mul_bwd_w13_{n_elems}", estimates=Estimates(ops=10*n_elems, mem=mem)))
  src, lib = compile_cpp(pathlib.Path(__file__).parent, "cast_amax_bwd_w13.cpp", n_elems, hidden, ["-DLAYER_SCALE=1"] if layer_num is not None else [])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def _custom_fused_cast_amax_w13(fp8_out:UOp, amax_out:UOp, xw13:UOp, amax_state:UOp, grad_amax_state:UOp, next_grad_amax_state:UOp, layer_num:UOp|None=None, dname:str=None) -> UOp:
  # NOTE: grad_amax_state is plumbed through as an unused fwd input so the bwd kernel can read it via kernel.src
  hidden = xw13.shape[2] // 2
  n_elems = xw13.shape[0] * xw13.shape[1] * hidden
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(NUM_WG, "gidx0")
  mem = n_elems * 2 * 2 + n_elems + 4
  if layer_num is None:
    sink = UOp.sink(fp8_out.base, amax_out, xw13.base, amax_state.base, threads, workgroups,
                    arg=KernelInfo(f"fused_silu_mul_cast_amax_w13_{n_elems}", estimates=Estimates(ops=5*n_elems, mem=mem)))
  else:
    sink = UOp.sink(fp8_out.base, amax_out.base, xw13.base, amax_state.base, grad_amax_state.base, next_grad_amax_state.base,
                    layer_num.base, threads, workgroups,
                    arg=KernelInfo(f"fused_silu_mul_cast_amax_w13_{n_elems}", estimates=Estimates(ops=5*n_elems, mem=mem)))
  src, lib = compile_cpp(pathlib.Path(__file__).parent, "cast_amax_fwd_w13.cpp", n_elems, hidden, ["-DLAYER_SCALE=1"] if layer_num is not None else [])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                                UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def _fused_quantize_bwd_w13(gradient:UOp, kernel:UOp):
  src = kernel.src[1:]
  _, _, xw13, amax_state, grad_amax_state, next_grad_amax_state = src[:6]
  layer_num = src[6] if len(src) > 6 else None
  device = xw13.device
  axis = xw13.axis if isinstance(device, tuple) else None
  grad_xw13_fp8 = alloc_like(xw13.shape, dtypes.fp8e4m3,  device, axis)
  grad_amax_next = Tensor(next_grad_amax_state, device=device) if layer_num is not None else zero_scalar(device)
  grad_amax_state_t = Tensor(grad_amax_state, device=device)
  fxn = functools.partial(_custom_fused_bwd_w13, dname=dname_of(device))
  grad_xw13_fp8, grad_amax_next, *_ = Tensor.custom_kernel(
    grad_xw13_fp8, grad_amax_next,
    Tensor(xw13, device=device), Tensor(gradient, device=device).cast(dtypes.bfloat16),
    Tensor(amax_state, device=device), grad_amax_state_t, *((Tensor(layer_num, device=device),) if layer_num is not None else ()), fxn=fxn)
  grad_xw13_uop = grad_xw13_fp8.uop.cast(dtypes.bfloat16)
  store_effect = None if layer_num is not None else next_grad_amax_state.store(grad_amax_next.uop)
  assert grad_xw13_fp8.uop.op is Ops.AFTER, f"expected AFTER, got {grad_xw13_fp8.uop.op}"
  grad_xw13_fp8_uop = grad_xw13_fp8.uop if store_effect is None else grad_xw13_fp8.uop.replace(src=grad_xw13_fp8.uop.src + (store_effect,))
  # Stash fp8 companion for cdna_asm_gemm's bwd to attach to grad_a.
  _grad_fp8_mailbox[grad_xw13_uop] = (grad_xw13_fp8_uop, grad_amax_state_t.uop)
  return (None, None, grad_xw13_uop, None, None, None) + ((None,) if layer_num is not None else ())

def fused_quantize_fp8_w13(xw13:Tensor, amax_state:Tensor, fp8_dtype, grad_amax_state:Tensor, next_grad_amax_state:Tensor,
                           amax_out:Tensor|None=None, layer_num:Tensor|None=None) -> tuple[Tensor, Tensor]:
  # NOTE: silu(xw1)*xw3 -> fp8 + amax over fused xw13 layout. Returns (fp8, new_amax)
  # grad_amax_state: delayed amax for grad_xw13 fp8 quantization in the backward.
  assert xw13.dtype == dtypes.bfloat16, f"expected bf16, got {xw13.dtype}"
  MBS, SEQ, H2 = xw13.shape
  assert H2 % 2 == 0, f"w13 last-axis must be even, got {H2}"
  HIDDEN = H2 // 2
  axis = xw13.uop.axis if isinstance(xw13.device, tuple) else None
  fp8_out  = alloc_like((MBS, SEQ, HIDDEN), fp8_dtype,      xw13.device, axis)
  amax_out = zero_scalar(xw13.device) if amax_out is None else amax_out
  fxn = functools.partial(_custom_fused_cast_amax_w13, dname=dname_of(xw13.device))
  fp8_out, amax_out, *_ = Tensor.custom_kernel(fp8_out, amax_out, xw13, amax_state, grad_amax_state, next_grad_amax_state, *((layer_num,) if layer_num is not None else ()),
                                                fxn=fxn, grad_fxn=_fused_quantize_bwd_w13)
  return fp8_out, amax_out
