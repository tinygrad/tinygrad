from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.compiler_amd import HIPCCCompiler

THREADS_PER_WG = 256

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

def _fused_ce_loss_bwd(gradient:UOp, kernel:UOp, label_smoothing:float):
  # NOTE: forward inputs are (loss_out, max_out, lse_out, logits, targets)
  # gradient is the upstream grad w.r.t. per-row loss (shape: (rows,) fp32)
  _, _, lse_u, logits_u, targets_u = kernel.src[1:]
  device = logits_u.device
  rows, VOCAB = logits_u.shape  # (rows, VOCAB) after reshape
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
  # NOTE: .mean() backward gives same grad per row (1/N), so broadcast is safe; take scalar
  scale = Tensor(gradient, device=device).float().reshape(-1)[0:1].contiguous()
  logits_t = Tensor(logits_u.after(kernel), device=device)
  lse_t = Tensor(lse_u.after(kernel), device=device)
  targets_t = Tensor(targets_u, device=device)
  fxn = functools.partial(_custom_fused_ce_loss_bwd, dname=dname, vocab=VOCAB, rows=rows_per_dev, label_smoothing=label_smoothing)
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
    fxn=fxn, grad_fxn=functools.partial(_fused_ce_loss_bwd, label_smoothing=label_smoothing))
  return loss_out.mean()
