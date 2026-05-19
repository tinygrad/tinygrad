from __future__ import annotations
import functools
from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import UOp, KernelInfo, AxisType

@functools.cache
def _fused_ce_loss_fwd(loss_out:UOp, max_out:UOp, lse_out:UOp, logits:UOp, targets:UOp,
                       vocab:int, rows:int, label_smoothing:float) -> UOp:
  row = UOp.range(rows, 0)
  target = targets[row].cast(dtypes.weakint)

  max_acc = UOp.placeholder((1,), dtypes.float, 0, addrspace=AddrSpace.REG)
  max_acc = max_acc.after(row)[0].set(float("-inf"))
  v_max = UOp.range(vocab, 1, axis_type=AxisType.REDUCE)
  x_max = logits[row, v_max].cast(dtypes.float)
  max_acc = max_acc[0].set(max_acc.after(v_max)[0].maximum(x_max), end=v_max)
  row_max = max_acc[0]

  sum_exp = UOp.placeholder((1,), dtypes.float, 1, addrspace=AddrSpace.REG)
  sum_exp = sum_exp.after(row)[0].set(0.0)
  sum_x = UOp.placeholder((1,), dtypes.float, 2, addrspace=AddrSpace.REG)
  sum_x = sum_x.after(row)[0].set(0.0)
  v_sum = UOp.range(vocab, 2, axis_type=AxisType.REDUCE)
  x_sum = logits[row, v_sum].cast(dtypes.float)
  sum_exp = sum_exp[0].set(sum_exp.after(v_sum)[0] + (x_sum - row_max).exp(), end=v_sum)
  v_x = UOp.range(vocab, 3, axis_type=AxisType.REDUCE)
  sum_x = sum_x[0].set(sum_x.after(v_x)[0] + logits[row, v_x].cast(dtypes.float), end=v_x)

  target_logit = logits[row, target].cast(dtypes.float)
  row_lse = sum_exp[0].log() + row_max
  loss = row_lse - (1.0 - label_smoothing) * target_logit - label_smoothing * (sum_x[0] / vocab)
  stores = UOp.group(loss_out[row].store(loss), max_out[row].store(row_max), lse_out[row].store(row_lse))
  # TODO: remove the need for this.
  return stores.end(row).sink(arg=KernelInfo(f"fused_ce_loss_fwd_{rows}_{vocab}", opts_to_apply=()))

@functools.cache
def _fused_ce_loss_bwd_kernel(d_logits:UOp, logits:UOp, lse:UOp, targets:UOp, scale:UOp,
                              vocab:int, rows:int, label_smoothing:float) -> UOp:
  row = UOp.range(rows, 0)
  v = UOp.range(vocab, 1)
  x = logits[row, v].cast(dtypes.float)
  prob = (x - lse[row]).exp()
  target_term = v.eq(targets[row].cast(dtypes.weakint)).where(1.0 - label_smoothing, 0.0)
  grad = (prob - target_term - label_smoothing / vocab) * scale[0]
  return d_logits[row, v].store(grad.cast(d_logits.dtype.base)).end(v, row).sink(arg=KernelInfo(f"fused_ce_loss_bwd_{rows}_{vocab}"))

def _fused_ce_loss_bwd(gradient:UOp, kernel:UOp, label_smoothing:float):
  _, _, lse_u, logits_u, targets_u = kernel.src[1:]
  device = logits_u.device
  rows, VOCAB = logits_u.shape
  if isinstance(device, tuple):
    axis = logits_u.axis
    ndev = len(device)
    d_logits = Tensor(Tensor.invalids(rows // ndev, VOCAB, dtype=dtypes.bfloat16, device=device).uop.multi(axis), device=device)
    rows_per_dev = rows // ndev
  else:
    d_logits = Tensor.invalids(rows, VOCAB, dtype=dtypes.bfloat16, device=device)
    rows_per_dev = rows
  scale = Tensor(gradient, device=device).float().reshape(-1)[0:1].contiguous()
  logits_t = Tensor(logits_u.after(kernel), device=device)
  lse_t = Tensor(lse_u.after(kernel), device=device)
  targets_t = Tensor(targets_u, device=device)
  fxn = functools.partial(_fused_ce_loss_bwd_kernel, vocab=VOCAB, rows=rows_per_dev, label_smoothing=label_smoothing)
  d_logits, *_ = Tensor.custom_kernel(d_logits, logits_t, lse_t, targets_t, scale, fxn=fxn)
  return (None, None, None, d_logits.uop, None)

def fused_ce_loss(logits:Tensor, targets:Tensor, label_smoothing:float=0.1) -> Tensor:
  # NOTE: fused sparse_categorical_crossentropy with label smoothing, returns mean loss scalar.
  assert logits.dtype == dtypes.bfloat16, f"expected bf16, got {logits.dtype}"
  assert logits.ndim == 3, f"expected (MBS, SEQ, VOCAB), got {logits.shape}"
  MBS, SEQ, VOCAB = logits.shape
  rows = MBS * SEQ
  if isinstance(logits.device, tuple):
    axis = logits.uop.axis
    assert axis in (0, 1), f"unsupported sharding axis={axis} for CE loss"
    ndev = len(logits.device)
    loss_out = Tensor(Tensor.invalids(rows // ndev, dtype=dtypes.float32, device=logits.device).uop.multi(0), device=logits.device)
    max_out  = Tensor(Tensor.invalids(rows // ndev, dtype=dtypes.float32, device=logits.device).uop.multi(0), device=logits.device)
    lse_out  = Tensor(Tensor.invalids(rows // ndev, dtype=dtypes.float32, device=logits.device).uop.multi(0), device=logits.device)
    rows_per_dev = rows // ndev
  else:
    loss_out = Tensor.invalids(rows, dtype=dtypes.float32, device=logits.device)
    max_out  = Tensor.invalids(rows, dtype=dtypes.float32, device=logits.device)
    lse_out  = Tensor.invalids(rows, dtype=dtypes.float32, device=logits.device)
    rows_per_dev = rows
  logits_flat = logits.reshape(rows, VOCAB)
  targets_flat = targets.reshape(-1).cast(dtypes.int32)
  fxn = functools.partial(_fused_ce_loss_fwd, vocab=VOCAB, rows=rows_per_dev, label_smoothing=label_smoothing)
  loss_out, max_out, lse_out, *_ = Tensor.custom_kernel(
    loss_out, max_out, lse_out, logits_flat, targets_flat,
    fxn=fxn, grad_fxn=functools.partial(_fused_ce_loss_bwd, label_smoothing=label_smoothing))
  return loss_out.mean()
