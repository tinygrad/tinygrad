from __future__ import annotations
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
  s = list(shape)
  s[axis] //= ndev
  return s

def _scalar_amax(amax_buf:Tensor) -> Tensor:
  if isinstance(amax_buf.device, tuple):
    from examples.mlperf.models.flat_llama import _local_abs_max
    return _local_abs_max(amax_buf).detach()
  return amax_buf.max().detach()


# **** fused silu*mul -> fp8 cast + amax (w13 layout) ****

@functools.cache
def _custom_fused_bwd_w13(grad_xw13:UOp, xw13:UOp, grad_x2:UOp, amax_state:UOp, dname:str) -> UOp:
  hidden = xw13.shape[2] // 2
  n_elems = xw13.shape[0] * xw13.shape[1] * hidden
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(NUM_WG, "gidx0")
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
  mem = n_elems * 2 * 2 + n_elems + NUM_WG * 2
  sink = UOp.sink(fp8_out.base, amax_buf.base, xw13.base, amax_state.base, threads, workgroups,
                  arg=KernelInfo(f"fused_silu_mul_cast_amax_w13_{n_elems}", estimates=Estimates(ops=5*n_elems, mem=mem)))
  src, lib = _compile("cast_amax_fwd_w13.cpp", n_elems, hidden)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def _fused_quantize_bwd_w13(gradient:UOp, kernel:UOp):
  # NOTE: inputs are (fp8_out, amax_buf, xw13, amax_state); grad for xw13 only
  _, _, xw13, amax_state = kernel.src[1:]
  device = xw13.device
  if isinstance(device, tuple):
    axis, ndev = xw13.axis, len(device)
    assert axis in (0, 1), f"unsupported sharding axis={axis}"
    grad_xw13 = Tensor(Tensor.invalids(*_shard_shape(xw13.shape, axis, ndev), dtype=dtypes.bfloat16,
                                        device=device).uop.multi(axis), device=device)
    dname = device[0].split(":")[0]
  else:
    grad_xw13 = Tensor.invalids(*xw13.shape, dtype=dtypes.bfloat16, device=device)
    dname = device.split(":")[0] if isinstance(device, str) else device
  grad_x2_t = Tensor(gradient, device=device).cast(dtypes.bfloat16)
  fxn = functools.partial(_custom_fused_bwd_w13, dname=dname)
  grad_xw13, *_ = Tensor.custom_kernel(grad_xw13, Tensor(xw13, device=device), grad_x2_t,
                                        Tensor(amax_state, device=device), fxn=fxn)
  return (None, None, grad_xw13.uop, None)

def fused_quantize_fp8_w13(xw13:Tensor, amax_state:Tensor, fp8_dtype) -> tuple[Tensor, Tensor, Tensor]:
  # NOTE: silu(xw1)*xw3 -> fp8 + amax over fused xw13 layout. Returns (fp8, inv_scale, new_amax)
  assert xw13.dtype == dtypes.bfloat16, f"expected bf16, got {xw13.dtype}"
  MBS, SEQ, H2 = xw13.shape
  assert H2 % 2 == 0, f"w13 last-axis must be even, got {H2}"
  HIDDEN = H2 // 2
  if isinstance(xw13.device, tuple):
    axis, ndev = xw13.uop.axis, len(xw13.device)
    assert axis in (0, 1), f"unsupported sharding axis={axis}"
    fp8_out = Tensor(Tensor.invalids(*_shard_shape((MBS, SEQ, HIDDEN), axis, ndev), dtype=fp8_dtype,
                                     device=xw13.device).uop.multi(axis), device=xw13.device)
    amax_buf = Tensor(Tensor.invalids(NUM_WG, dtype=dtypes.bfloat16, device=xw13.device).uop.multi(0),
                      device=xw13.device)
    dname = xw13.device[0].split(":")[0]
  else:
    fp8_out = Tensor.invalids(MBS, SEQ, HIDDEN, dtype=fp8_dtype, device=xw13.device)
    amax_buf = Tensor.invalids(NUM_WG, dtype=dtypes.bfloat16, device=xw13.device)
    dname = xw13.device.split(":")[0] if isinstance(xw13.device, str) else xw13.device
  fxn = functools.partial(_custom_fused_cast_amax_w13, dname=dname)
  fp8_out, amax_buf, *_ = Tensor.custom_kernel(fp8_out, amax_buf, xw13, amax_state, fxn=fxn,
                                                grad_fxn=_fused_quantize_bwd_w13)
  inv_scale = (amax_state.float() + 1e-8) / FP8_MAX
  return fp8_out, inv_scale, _scalar_amax(amax_buf)

# **** fused (x * weight) -> fp8 cast + amax (norm-mul-quantize) ****

@functools.cache
def _custom_mul_quantize_fp8(fp8_out:UOp, amax_buf:UOp, x:UOp, weight:UOp, amax_state:UOp, dname:str) -> UOp:
  MBS, SEQ, HIDDEN = x.shape
  n_elems = MBS * SEQ * HIDDEN
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(NUM_WG, "gidx0")
  mem = n_elems * 2 + HIDDEN * 2 + n_elems + NUM_WG * 2
  sink = UOp.sink(fp8_out.base, amax_buf.base, x.base, weight.base, amax_state.base, threads, workgroups,
                  arg=KernelInfo(f"fused_mul_quantize_fp8_{n_elems}_h{HIDDEN}", estimates=Estimates(ops=3*n_elems, mem=mem)))
  src, lib = _compile("fused_mul_quantize_fp8.cpp", n_elems, HIDDEN)
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
    fp8_out = Tensor(Tensor.invalids(*_shard_shape((MBS, SEQ, HIDDEN), axis, ndev), dtype=fp8_dtype,
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
  new_amax = _scalar_amax(amax_buf)
  inv_scale = (amax_state.float() + 1e-8) / FP8_MAX
  return fp8_out, inv_scale, new_amax

# **** fused ce loss ****

@functools.cache
def _custom_fused_ce_loss_fwd(loss_out:UOp, max_out:UOp, lse_out:UOp, logits:UOp, targets:UOp,
                              dname:str, vocab:int, rows:int, label_smoothing:float) -> UOp:
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(rows, "gidx0")
  mem = rows * vocab * 2 + rows * 12 + rows * 4
  sink = UOp.sink(loss_out.base, max_out.base, lse_out.base, logits.base, targets.base,
                  threads, workgroups,
                  arg=KernelInfo(f"fused_ce_loss_fwd", estimates=Estimates(ops=6*rows*vocab, mem=mem)))
  src = (pathlib.Path(__file__).parent/"fused_ce_loss.cpp").read_text()
  defines = [f"-DVOCAB={vocab}", f"-DTHREADS_PER_WG={THREADS_PER_WG}",
             f"-DLABEL_SMOOTHING={label_smoothing}f"]
  lib = HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", *defines]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

@functools.cache
def _custom_fused_ce_loss_bwd(d_logits:UOp, logits:UOp, lse:UOp, targets:UOp, scale:UOp,
                              dname:str, vocab:int, rows:int, label_smoothing:float) -> UOp:
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(rows, "gidx0")
  mem = rows * vocab * 4 + rows * 8 + 4
  sink = UOp.sink(d_logits.base, logits.base, lse.base, targets.base, scale.base,
                  threads, workgroups,
                  arg=KernelInfo(f"fused_ce_loss_bwd", estimates=Estimates(ops=4*rows*vocab, mem=mem)))
  src = (pathlib.Path(__file__).parent/"fused_ce_loss_bwd.cpp").read_text()
  defines = [f"-DVOCAB={vocab}", f"-DTHREADS_PER_WG={THREADS_PER_WG}",
             f"-DLABEL_SMOOTHING={label_smoothing}f"]
  lib = HIPCCCompiler("gfx950", ["-std=c++20", "-ffast-math", *defines]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def _fused_ce_loss_bwd(gradient:UOp, kernel:UOp):
  # NOTE: forward inputs are (loss_out, max_out, lse_out, logits, targets)
  # gradient is the upstream grad w.r.t. per-row loss (shape: (rows,) fp32)
  _, _, lse_u, logits_u, targets_u = kernel.src[1:]
  device = logits_u.device
  rows_vocab = logits_u.shape  # (rows, VOCAB) after reshape
  rows, VOCAB = rows_vocab
  if isinstance(device, tuple):
    axis = logits_u.axis
    ndev = len(device)
    d_logits = Tensor(Tensor.invalids(rows // ndev, VOCAB, dtype=dtypes.bfloat16, device=device).uop.multi(axis), device=device)
    dname = device[0].split(":")[0]
    rows_per_dev = rows // ndev
  else:
    d_logits = Tensor.invalids(rows, VOCAB, dtype=dtypes.bfloat16, device=device)
    dname = device.split(":")[0] if isinstance(device, str) else device
    rows_per_dev = rows
  grad_t = Tensor(gradient, device=device).float().reshape(-1)  # (rows,) fp32
  # NOTE: .mean() backward gives same grad per row (1/N), so broadcast is safe; take scalar
  scale = grad_t[0:1].contiguous()
  logits_t = Tensor(logits_u.after(kernel), device=device)
  lse_t = Tensor(lse_u.after(kernel), device=device)
  targets_t = Tensor(targets_u, device=device)
  fxn = functools.partial(_custom_fused_ce_loss_bwd, dname=dname, vocab=VOCAB, rows=rows_per_dev, label_smoothing=0.1)
  d_logits, *_ = Tensor.custom_kernel(d_logits, logits_t, lse_t, targets_t, scale, fxn=fxn)
  return (None, None, None, d_logits.uop, None)

def fused_ce_loss(logits:Tensor, targets:Tensor, label_smoothing:float=0.1) -> Tensor:
  # NOTE: fused sparse_categorical_crossentropy with label smoothing, returns mean loss scalar
  assert logits.dtype == dtypes.bfloat16, f"expected bf16, got {logits.dtype}"
  assert logits.ndim == 3, f"expected (MBS, SEQ, VOCAB), got {logits.shape}"
  MBS, SEQ, VOCAB = logits.shape
  rows = MBS * SEQ
  if isinstance(logits.device, tuple):
    axis = logits.uop.axis
    assert axis in (0, 1), f"unsupported sharding axis={axis} for CE loss"
    ndev = len(logits.device)
    loss_out = Tensor(Tensor.invalids(rows // ndev, dtype=dtypes.float32, device=logits.device).uop.multi(0),
                      device=logits.device)
    max_out  = Tensor(Tensor.invalids(rows // ndev, dtype=dtypes.float32, device=logits.device).uop.multi(0),
                      device=logits.device)
    lse_out  = Tensor(Tensor.invalids(rows // ndev, dtype=dtypes.float32, device=logits.device).uop.multi(0),
                      device=logits.device)
    dname = logits.device[0].split(":")[0]
    rows_per_dev = rows // ndev
  else:
    loss_out = Tensor.invalids(rows, dtype=dtypes.float32, device=logits.device)
    max_out  = Tensor.invalids(rows, dtype=dtypes.float32, device=logits.device)
    lse_out  = Tensor.invalids(rows, dtype=dtypes.float32, device=logits.device)
    dname = logits.device.split(":")[0] if isinstance(logits.device, str) else logits.device
    rows_per_dev = rows
  logits_flat = logits.reshape(rows, VOCAB)
  targets_flat = targets.reshape(-1).cast(dtypes.int32)
  fxn = functools.partial(_custom_fused_ce_loss_fwd, dname=dname, vocab=VOCAB, rows=rows_per_dev,
                          label_smoothing=label_smoothing)
  loss_out, max_out, lse_out, *_ = Tensor.custom_kernel(
    loss_out, max_out, lse_out, logits_flat, targets_flat,
    fxn=fxn, grad_fxn=_fused_ce_loss_bwd)
  return loss_out.mean()

