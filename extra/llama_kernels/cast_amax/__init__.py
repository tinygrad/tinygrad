from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import ContextVar, prod
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.renderer import Estimates
from extra.llama_kernels import FP8_MAX, NUM_WG, THREADS_PER_WG, compile_cpp, alloc_like, alloc_local, scalar_amax, dname_of

# module-level mailbox: grad_xw13 UOp -> (grad_xw13_fp8 UOp, inv_scale UOp, new_amax UOp, store_effect)
# lets cdna_asm_gemm's bwd reuse the fp8 companion produced by the fused silu_mul bwd kernel
# instead of doing a redundant bf16 -> fp8 quantize.
_grad_fp8_mailbox:dict = {}

@functools.cache
def _custom_fused_bwd_w13(grad_xw13:UOp, grad_xw13_fp8:UOp, grad_amax_buf:UOp,
                          xw13:UOp, grad_x2:UOp, amax_state:UOp, grad_amax_state:UOp, dname:str) -> UOp:
  hidden = xw13.shape[2] // 2
  n_elems = xw13.shape[0] * xw13.shape[1] * hidden
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(NUM_WG, "gidx0")
  mem = n_elems * 2 * 5 + n_elems * 2 + NUM_WG * 4 + 4
  sink = UOp.sink(grad_xw13.base, grad_xw13_fp8.base, grad_amax_buf.base,
                  xw13.base, grad_x2.base, amax_state.base, grad_amax_state.base, threads, workgroups,
                  arg=KernelInfo(f"fused_silu_mul_bwd_w13_{n_elems}", estimates=Estimates(ops=10*n_elems, mem=mem)))
  src, lib = compile_cpp(pathlib.Path(__file__).parent, "cast_amax_bwd_w13.cpp", n_elems, hidden)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def _custom_fused_cast_amax_w13_uop(fp8_out:UOp, amax_buf:UOp, xw13:UOp, amax_state:UOp, grad_amax_state:UOp) -> UOp:
  VEC = 8
  n_elems = prod(fp8_out.shape)
  hidden = fp8_out.shape[-1] if len(fp8_out.shape) > 1 else xw13.shape[-1] // 2
  hidden_vec = hidden // VEC

  assert n_elems % (NUM_WG * THREADS_PER_WG * VEC) == 0
  assert hidden % VEC == 0
  assert prod(xw13.shape) == 2 * n_elems
  assert amax_buf.shape[0] == NUM_WG

  fp8_out = fp8_out.reshape(n_elems)
  xw13 = xw13.reshape(n_elems // hidden, 2, hidden)

  wg = UOp.range(NUM_WG, 0, AxisType.GLOBAL)
  tid = UOp.range(THREADS_PER_WG, 1, AxisType.LOCAL)
  it = UOp.range((n_elems // VEC) // (NUM_WG * THREADS_PER_WG), 2, AxisType.LOOP)
  lane = UOp.range(VEC, 3, AxisType.UNROLL)

  vec = (it * NUM_WG + wg) * THREADS_PER_WG + tid
  idx = vec * VEC + lane
  row = vec // hidden_vec
  col = (vec % hidden_vec) * VEC + lane

  scale = FP8_MAX / (amax_state[0].cast(dtypes.float) + 1e-8)

  f1 = xw13[row, 0, col].cast(dtypes.float)
  f3 = xw13[row, 1, col].cast(dtypes.float)
  x2 = (f1 * (1.0 + (-f1).exp()).reciprocal()) * f3

  abs_x2 = x2.maximum(-x2)
  scaled = (x2 * scale).maximum(-FP8_MAX).minimum(FP8_MAX)

  fp8_store = fp8_out[idx].store(scaled.cast(fp8_out.dtype.base)).end(lane)
  lane_max = abs_x2.reduce(lane, arg=Ops.MAX)

  lmax = UOp.placeholder((1,), dtypes.float, slot=1, addrspace=AddrSpace.REG)
  lmax_init = lmax.after(wg, tid)[0].store(0.0)
  lmax_prev = lmax.after(lmax_init, it)[0]
  lmax_store = lmax.after(fp8_store)[0].store(lmax_prev.maximum(lane_max))
  lmax_val = lmax.after(lmax_store.end(it))[0]

  lds = UOp.placeholder((THREADS_PER_WG,), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL)
  lds = lds.after(lds[tid].store(lmax_val).barrier())

  step = THREADS_PER_WG // 2
  while step:
    active = tid < step
    other = lds[tid + step].load(UOp.const(dtypes.float, 0.0), active)
    lds = lds.after(lds[tid].store(lds[tid].maximum(other), gate=active).barrier())
    step //= 2

  amax_store = amax_buf[tid.eq(0).where(wg, UOp.invalid())].store(lds[0])
  return amax_store.end(tid, wg).sink(arg=KernelInfo(f"fused_silu_mul_cast_amax_w13_{n_elems}", opts_to_apply=()))

def _fused_quantize_bwd_w13(gradient:UOp, kernel:UOp):
  _, _, xw13, amax_state, grad_amax_state = kernel.src[1:]
  device = xw13.device
  axis = xw13.axis if isinstance(device, tuple) else None
  grad_xw13     = alloc_like(xw13.shape, dtypes.bfloat16, device, axis)
  grad_xw13_fp8 = alloc_like(xw13.shape, dtypes.fp8e4m3,  device, axis)
  grad_amax_buf = alloc_local((NUM_WG,), dtypes.float32,  device, axis)
  grad_amax_state_t = Tensor(grad_amax_state, device=device)
  fxn = functools.partial(_custom_fused_bwd_w13, dname=dname_of(device))
  grad_xw13, grad_xw13_fp8, grad_amax_buf, *_ = Tensor.custom_kernel(
    grad_xw13, grad_xw13_fp8, grad_amax_buf,
    Tensor(xw13, device=device), Tensor(gradient, device=device).cast(dtypes.bfloat16),
    Tensor(amax_state, device=device), grad_amax_state_t, fxn=fxn)
  inv_scale = (grad_amax_state_t.float() + 1e-8) / FP8_MAX
  new_grad_amax = scalar_amax(grad_amax_buf)
  store_effect = grad_amax_state_t.uop.store(new_grad_amax.uop)
  assert grad_xw13_fp8.uop.op is Ops.AFTER, f"expected AFTER, got {grad_xw13_fp8.uop.op}"
  grad_xw13_fp8_uop = grad_xw13_fp8.uop.replace(src=grad_xw13_fp8.uop.src + (store_effect,))
  # Stash fp8 companion for cdna_asm_gemm's bwd to attach to grad_a.
  _grad_fp8_mailbox[grad_xw13.uop] = (grad_xw13_fp8_uop, inv_scale.uop)
  return (None, None, grad_xw13.uop, None, None)

def fused_quantize_fp8_w13(xw13:Tensor, amax_state:Tensor, fp8_dtype, grad_amax_state:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  # NOTE: silu(xw1)*xw3 -> fp8 + amax over fused xw13 layout. Returns (fp8, inv_scale, new_amax)
  # grad_amax_state: delayed amax for grad_xw13 fp8 quantization in the backward.
  assert xw13.dtype == dtypes.bfloat16, f"expected bf16, got {xw13.dtype}"
  MBS, SEQ, H2 = xw13.shape
  assert H2 % 2 == 0, f"w13 last-axis must be even, got {H2}"
  HIDDEN = H2 // 2
  axis = xw13.uop.axis if isinstance(xw13.device, tuple) else None
  fp8_out  = alloc_like((MBS, SEQ, HIDDEN), fp8_dtype,      xw13.device, axis)
  amax_buf = alloc_local((NUM_WG,),         dtypes.float32, xw13.device, axis)
  fxn = _custom_fused_cast_amax_w13_uop
  fp8_out, amax_buf, *_ = Tensor.custom_kernel(fp8_out, amax_buf, xw13, amax_state, grad_amax_state,
                                                fxn=fxn, grad_fxn=_fused_quantize_bwd_w13)
  inv_scale = (amax_state.float() + 1e-8) / FP8_MAX
  return fp8_out, inv_scale, scalar_amax(amax_buf)
