import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.compiler_amd import HIPCCCompiler

FP8_MAX = 448.0
NUM_WG, THREADS_PER_WG = 1024, 256

def _compile(cpp_name:str, n_elems:int, hidden:int):
  src = (pathlib.Path(__file__).parent/cpp_name).read_text()
  defines = [f"-DN_ELEMS={n_elems}", f"-DHIDDEN={hidden}", f"-DNUM_WG={NUM_WG}", f"-DTHREADS_PER_WG={THREADS_PER_WG}"]
  return src, HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", *defines]).compile_cached(src)

def _shard_shape(shape:tuple, axis:int, ndev:int) -> list:
  s = list(shape); s[axis] //= ndev; return s

@functools.cache
def _custom_fused_bwd_w13(grad_xw13:UOp, xw13:UOp, grad_x2:UOp, amax_state:UOp, dname:str) -> UOp:
  hidden = xw13.shape[2] // 2
  n_elems = xw13.shape[0] * xw13.shape[1] * hidden
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(NUM_WG, "gidx0")
  # read 2*N bf16 (xw13) + N bf16 (grad_x2) + 1 scalar; write 2*N bf16 (grad_xw13)
  mem = n_elems * 2 * 5
  sink = UOp.sink(grad_xw13.base, xw13.base, grad_x2.base, amax_state.base, threads, workgroups,
                  arg=KernelInfo(f"fused_silu_mul_bwd_w13_{n_elems}", estimates=Estimates(ops=8*n_elems, mem=mem)))
  src, lib = _compile("cast_amax_bwd_w13.cpp", n_elems, hidden)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def _custom_fused_cast_amax_w13(fp8_out:UOp, amax_buf:UOp, xw13:UOp, amax_state:UOp, dname:str) -> UOp:
  hidden = xw13.shape[2] // 2
  n_elems = xw13.shape[0] * xw13.shape[1] * hidden
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(NUM_WG, "gidx0")
  # read 2*N bf16 + 1 scalar, write N fp8 + NUM_WG bf16
  mem = n_elems * 2 * 2 + n_elems + NUM_WG * 2
  sink = UOp.sink(fp8_out.base, amax_buf.base, xw13.base, amax_state.base, threads, workgroups,
                  arg=KernelInfo(f"fused_silu_mul_cast_amax_w13_{n_elems}", estimates=Estimates(ops=5*n_elems, mem=mem)))
  src, lib = _compile("cast_amax_fwd_w13.cpp", n_elems, hidden)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def _fused_quantize_bwd_w13(gradient:UOp, kernel:UOp):
  # kernel.src[1:] is (fp8_out, amax_buf, xw13, amax_state); only xw13 needs a grad
  _, _, xw13, amax_state = kernel.src[1:]
  device = xw13.device
  if isinstance(device, tuple):
    axis, ndev = xw13.axis, len(device)
    assert axis in (0, 1), f"unsupported sharding axis={axis}"
    grad_xw13 = Tensor(Tensor.invalid(*_shard_shape(xw13.shape, axis, ndev), dtype=dtypes.bfloat16, device=device).uop.multi(axis), device=device)
    dname = device[0].split(":")[0]
  else:
    grad_xw13 = Tensor.invalid(*xw13.shape, dtype=dtypes.bfloat16, device=device)
    dname = device.split(":")[0] if isinstance(device, str) else device
  grad_x2_t = Tensor(gradient, device=device).cast(dtypes.bfloat16)
  fxn = functools.partial(_custom_fused_bwd_w13, dname=dname)
  grad_xw13, *_ = Tensor.custom_kernel(grad_xw13, Tensor(xw13, device=device), grad_x2_t, Tensor(amax_state, device=device), fxn=fxn)
  return (None, None, grad_xw13.uop, None)

def fused_quantize_fp8_w13(xw13:Tensor, amax_state:Tensor, fp8_dtype) -> tuple[Tensor, Tensor, Tensor]:
  # silu(xw1)*xw3 -> fp8 + amax over fused xw13 layout. Returns (fp8, inv_scale, new_amax).
  assert xw13.dtype == dtypes.bfloat16, f"expected bf16, got {xw13.dtype}"
  MBS, SEQ, H2 = xw13.shape
  assert H2 % 2 == 0, f"w13 last-axis must be even, got {H2}"
  HIDDEN = H2 // 2
  if isinstance(xw13.device, tuple):
    axis, ndev = xw13.uop.axis, len(xw13.device)
    assert axis in (0, 1), f"unsupported sharding axis={axis}"
    fp8_out = Tensor(Tensor.invalid(*_shard_shape((MBS, SEQ, HIDDEN), axis, ndev), dtype=fp8_dtype, device=xw13.device).uop.multi(axis), device=xw13.device)
    amax_buf = Tensor(Tensor.invalid(NUM_WG, dtype=dtypes.bfloat16, device=xw13.device).uop.multi(0), device=xw13.device)
    dname = xw13.device[0].split(":")[0]
  else:
    fp8_out = Tensor.invalid(MBS, SEQ, HIDDEN, dtype=fp8_dtype, device=xw13.device)
    amax_buf = Tensor.invalid(NUM_WG, dtype=dtypes.bfloat16, device=xw13.device)
    dname = xw13.device.split(":")[0] if isinstance(xw13.device, str) else xw13.device
  fxn = functools.partial(_custom_fused_cast_amax_w13, dname=dname)
  fp8_out, amax_buf, *_ = Tensor.custom_kernel(fp8_out, amax_buf, xw13, amax_state, fxn=fxn, grad_fxn=_fused_quantize_bwd_w13)
  # per-device scalar amax (no cross-device allreduce, matches _local_abs_max semantics)
  if isinstance(amax_buf.device, tuple):
    from examples.mlperf.models.flat_llama import _local_abs_max
    new_amax = _local_abs_max(amax_buf).detach()
  else: new_amax = amax_buf.max().detach()
  inv_scale = (FP8_MAX / (amax_state + 1e-8)).float().reciprocal()
  return fp8_out, inv_scale, new_amax
