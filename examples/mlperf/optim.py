import functools
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.optim import Optimizer, OptimizerGroup
from tinygrad.helpers import FUSE_OPTIM, getenv
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo

STOCHASTIC_ROUND = getenv("STOCHASTIC_ROUND", 0)
MASTER_WEIGHTS = getenv("MASTER_WEIGHTS", 0)
ZERO_OPTIM = getenv("ZERO_OPTIM", 0)
FP8_AMAX_MARGIN = getenv("FP8_AMAX_MARGIN", 1.1)
IMMEDIATE_SCALE = getenv("IMMEDIATE_SCALE", 0)
MXFP8 = getenv("MXFP8", 0)

def stochastic_round_bf16(x:Tensor) -> Tensor:
  bits = x.bitcast(dtypes.uint32)
  if isinstance(x.device, tuple):
    shape = x.uop.shard_shape if x.uop.axis is not None else x.shape
    noise = Tensor(UOp(Ops.MSTACK, dtypes.default_float, tuple(Tensor.rand(*shape, device=d).uop for d in x.device)))
  else:
    noise = x.rand_like()
  noise = (noise * 0xFFFF).cast(dtypes.uint32)
  return ((bits + noise) & 0xFFFF0000).bitcast(dtypes.float32).cast(dtypes.bfloat16)

def clip_grads(grads:list[Tensor], grad_acc, clip_norm) -> tuple[Tensor, Tensor]:
  # Match the BF16 rounding of the former in-place divide while leaving gradients untouched for the optimizer.
  avg_grads = [(g / grad_acc).cast(g.dtype) for g in grads]
  device = avg_grads[0].device
  if isinstance(device, tuple) and all(g.device == device and g.uop.axis is None for g in avg_grads):
    # Replicated gradients have identical storage on every device. Partition divisible tensors across the devices so
    # each element contributes once; for indivisible tensors, split the identical local sum evenly instead.
    n, dnum = len(device), UOp.range(len(device), -1, AxisType.DEVICE)
    local_sq = [Tensor(g.uop._shard(0, dnum)).float().square().sum() if g.ndim and g.shape[0] % n == 0 else
                g.float().square().sum() / n for g in avg_grads]
    local_norm = Tensor.stack(*local_sq).sum()
    total_norm = Tensor(local_norm.uop.allreduce(Ops.ADD, device)).sqrt().contiguous()
  else:
    total_norm = Tensor.stack(*[g.float().square().sum() for g in avg_grads]).sum().sqrt().contiguous()
  return total_norm, (clip_norm / (total_norm + 1e-6)).clamp(max_=1.0).contiguous()

@functools.cache
def _adamw_master_kernel(m:UOp, v:UOp, master:UOp, param:UOp, grad:UOp, lr:UOp, b1_t:UOp, b2_t:UOp, clip_coeff:UOp,
                         *, b1:float, b2:float, eps:float, wd:float, grad_acc:int) -> UOp:
  m, v, master, param, grad = (x.flatten() for x in (m, v, master, param, grad))
  assert m.shape == v.shape == master.shape == param.shape and grad.numel() % m.numel() == 0
  idx = UOp.range(m.numel(), 0)
  grad_idx = idx if grad.numel() == m.numel() else idx + UOp.range(grad.numel() // m.numel(), -1, AxisType.DEVICE) * m.numel()
  # Preserve the two BF16 rounding points from clip_grads: divide, then multiply by the clip coefficient.
  g = (grad[grad_idx] / grad_acc).cast(grad.dtype)
  g = (g * clip_coeff.flatten()[0]).cast(grad.dtype).cast(dtypes.float32)
  old_m, old_v, old_w = m[idx].cast(dtypes.float32), v[idx].cast(dtypes.float32), master[idx].cast(dtypes.float32)
  new_m = b1 * old_m + (1.0 - b1) * g
  new_v = b2 * old_v + (1.0 - b2) * g * g
  update = (new_m / (1.0 - b1_t.flatten()[0])) / ((new_v / (1.0 - b2_t.flatten()[0])).sqrt() + eps)
  new_w = old_w - lr.flatten()[0] * (update + wd * old_w)
  stores = (m[idx].store(new_m.cast(m.dtype)), v[idx].store(new_v.cast(v.dtype)), master[idx].store(new_w.cast(master.dtype)),
            param[idx].store(new_w.cast(param.dtype)))
  return UOp.group(*stores).end(idx).sink(arg=KernelInfo(f"adamw_master_{m.numel()}"))

def _adamw_master_step(param:Tensor, grad:Tensor, m:Tensor, v:Tensor, master:Tensor, lr:Tensor, b1_t:Tensor, b2_t:Tensor,
                       *, b1:float, b2:float, eps:float, wd:float, clip_coeff:Tensor|None=None, grad_acc:int=1) -> None:
  if clip_coeff is None: clip_coeff = Tensor.ones(1, dtype=grad.dtype, device=grad.device).contiguous()
  fxn = functools.partial(_adamw_master_kernel, b1=b1, b2=b2, eps=eps, wd=wd, grad_acc=grad_acc)
  updated = Tensor.custom_kernel(m, v, master, param, grad, lr, b1_t, b2_t, clip_coeff, fxn=fxn)
  for dst, src in zip((m, v, master, param), updated): dst.replace(src)

class GradAccClipAdamW(Optimizer):
  def __init__(self, params:list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.0, grad_acc=1, clip_norm=1.0, device=None, fused=FUSE_OPTIM):
    super().__init__(params, lr, device, fused)
    self.b1, self.b2, self.eps, self.wd = b1, b2, eps, weight_decay
    self.b1_t, self.b2_t = (Tensor.ones((1,), dtype=dtypes.float32, device=self.device) for _ in [b1, b2])
    self.zero = bool(ZERO_OPTIM) and isinstance(self.device, tuple) and not self.fused
    self.m = [self._zero_shard(x) for x in self._new_optim_param()]
    self.v = [self._zero_shard(x) for x in self._new_optim_param()]
    self.grad_acc, self.clip_norm = grad_acc, clip_norm
    if MASTER_WEIGHTS and self.params[0].dtype != dtypes.float32:
      self.master_params:list[Tensor]|None = [self._zero_shard(p.to(self.device).float().contiguous()) for p in self.params]
    else:
      self.master_params = None
    self.param_shards = [self._zero_shard(p) for p in self.params] if self.zero else self.params

  def _zero_shard(self, t:Tensor) -> Tensor:
    if not self.zero or t.ndim < 2 or (t.shape[0] % len(self.device)) != 0: return t
    return Tensor(t.uop._shard(0, UOp.range(len(self.device), -1, AxisType.DEVICE)).unshard(0)).clone()

  def _zero_gather(self, t:Tensor) -> Tensor:
    if not isinstance(t.device, tuple) or t.uop.axis != 0: return t
    n, sz = len(t.device), t.shape[0] // len(t.device)
    return Tensor.cat(*[t[p*sz:(p+1)*sz] for p in range(n)], dim=0)

  def fschedule_step(self, grads:list[Tensor], clip_coeff:Tensor) -> list[Tensor]:
    if self.master_params is not None and not STOCHASTIC_ROUND and all(
      p.dtype == g.dtype == dtypes.bfloat16 and m.dtype in (dtypes.bfloat16, dtypes.float32) and v.dtype == m.dtype and
      master.dtype == dtypes.float32 and p.device == g.device == m.device == v.device == master.device
      for p, g, m, v, master in zip(self.params, grads, self.m, self.v, self.master_params)
    ):
      self.b1_t *= self.b1
      self.b2_t *= self.b2
      for p, p_shard, g, m, v, master in zip(self.params, self.param_shards, grads, self.m, self.v, self.master_params):
        _adamw_master_step(p_shard, g, m, v, master, self.lr, self.b1_t, self.b2_t, b1=self.b1, b2=self.b2, eps=self.eps,
                           wd=self.wd, clip_coeff=clip_coeff, grad_acc=self.grad_acc)
        if p_shard is not p: p.assign(self._zero_gather(p_shard))
      return [self.b1_t, self.b2_t] + self.m + self.v + self.params + self.master_params
    grads = [((g / self.grad_acc).cast(g.dtype) * clip_coeff).cast(g.dtype) for g in grads]
    updates, extra = self._step([], grads)
    for i, tt in enumerate(self.params): tt.assign(self._apply_update(tt, updates[i], self.master_params[i] if self.master_params else None))
    fp8_inv_scales = [tt._inv_scale for tt in self.params if hasattr(tt, '_inv_scale')]
    fp8_next_inv_scales = [tt._next_inv_scale for tt in self.params if hasattr(tt, '_next_inv_scale')]
    return extra + self.params + self.buffers + (self.master_params or []) + fp8_inv_scales + fp8_next_inv_scales

  def fstep(self, grads:list[Tensor], grad_norm:Tensor, clip_coeff:Tensor):
    Tensor.realize(grad_norm, *self.fschedule_step(grads, clip_coeff))

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    grads = list(grads)

    for i in range(len(grads)):
      if grads[i].device != self.m[i].device: grads[i] = grads[i].to(self.m[i].device)
    ret = []
    self.b1_t *= self.b1
    self.b2_t *= self.b2
    for i, g in enumerate(grads):
      m_new = self.b1 * self.m[i].float() + (1.0 - self.b1) * g.float()
      v_new = self.b2 * self.v[i].float() + (1.0 - self.b2) * (g.float() * g.float())
      self.m[i].assign(m_new.cast(self.m[i].dtype))
      self.v[i].assign(v_new.cast(self.v[i].dtype))
      m_hat = m_new / (1.0 - self.b1_t)
      v_hat = v_new / (1.0 - self.b2_t)
      up = m_hat / (v_hat.sqrt() + self.eps)
      ret.append(self.lr * up)
    return ret, [self.b1_t, self.b2_t] + self.m + self.v

  def _apply_update(self, t:Tensor, up:Tensor, master:Tensor|None=None) -> Tensor:
    w = master if master is not None else t
    up = up.float().shard_like(w) + self.lr.to(w.device) * self.wd * w.detach()
    new_w = w.detach() - up
    if master is not None: master.assign(new_w)
    if self.zero and not (MXFP8 and t.dtype in dtypes.fp8s): new_w = self._zero_gather(new_w)
    # when master is offloaded to a different device than the param, results are resharded back onto the param's (sharded) device
    offloaded = master is not None and master.device != t.device
    if STOCHASTIC_ROUND and t.dtype == dtypes.bfloat16:
      out = stochastic_round_bf16(new_w)
      return out.shard_like(t) if offloaded else out
    if t.dtype in dtypes.fp8s:
      if MXFP8:
        from extra.gemm.cdna_asm_gemm import quantize_mxfp8
        w_q, w_e8, _ = quantize_mxfp8(new_w.reshape(-1, new_w.shape[-1]))
        if self.zero: w_q, w_e8 = self._zero_gather(w_q), self._zero_gather(w_e8)
        new_e8 = w_e8.reshape(t._inv_scale.shape)
        t._inv_scale.assign(new_e8.shard_like(t._inv_scale) if offloaded else new_e8)
        ret = w_q.reshape(t.shape)
        return ret.shard_like(t) if offloaded else ret
      from examples.mlperf.models.flat_llama import FP8_MAX
      if IMMEDIATE_SCALE:
        amax_axis = tuple(range(t._inv_scale.ndim, new_w.ndim))
        new_inv = ((new_w.float().abs().max(axis=amax_axis).detach() + 1e-8) / FP8_MAX).cast(t._inv_scale.dtype)
        t._inv_scale.assign(new_inv.shard_like(t._inv_scale) if offloaded else new_inv)
        scale = new_inv.reciprocal().reshape(*new_inv.shape, *([1]*(new_w.ndim-new_inv.ndim)))
        ret = (new_w * scale).clamp(-FP8_MAX, FP8_MAX).cast(t.dtype)
        return ret.shard_like(t) if offloaded else ret
      # delayed scaling: reuse previous step's inv_scale
      t._inv_scale.assign(t._next_inv_scale)
      inv_scale = t._inv_scale.to(new_w.device) if offloaded else t._inv_scale
      scale = inv_scale.reciprocal().reshape(*inv_scale.shape, *([1]*(new_w.ndim-inv_scale.ndim)))
      scaled = (new_w * scale).clamp(-FP8_MAX, FP8_MAX)
      ret = scaled.cast(t.dtype)
      # update inv_scale for next step from quantized result
      new_amax = (ret.float().abs().max(axis=tuple(range(inv_scale.ndim, ret.ndim))) * inv_scale * FP8_AMAX_MARGIN).detach()
      new_inv = ((new_amax + 1e-8) / FP8_MAX).cast(t._inv_scale.dtype)
      t._next_inv_scale.assign(new_inv.shard_like(t._next_inv_scale) if offloaded else new_inv)
      return ret.shard_like(t) if offloaded else ret
    out = new_w.cast(t.dtype)
    return out.shard_like(t) if offloaded else out

class GradAccClipAdamWGroup(OptimizerGroup):
  def __init__(self, *optimizers:GradAccClipAdamW):
    super().__init__(*optimizers)
    for o in self.optimizers[1:]: o.lr = self.optimizers[0].lr
  def fstep(self, grads:list[Tensor], grad_norm:Tensor, clip_coeff:Tensor):
    offset = 0
    to_realize = []
    for o in self.optimizers:
      n = len(o.params)
      to_realize += o.fschedule_step(grads[offset:offset+n], clip_coeff)
      offset += n
    Tensor.realize(*to_realize, grad_norm)
  @property
  def lr(self): return self.optimizers[0].lr
  @property
  def device(self): return self.optimizers[0].device
  @property
  def master_params(self):
    mp = [mp for o in self.optimizers for mp in (o.master_params or [])]
    return mp if mp else None
