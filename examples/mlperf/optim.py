from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.nn.optim import Optimizer, OptimizerGroup
from tinygrad.helpers import FUSE_OPTIM, getenv
from tinygrad.uop.ops import UOp, Ops, AxisType

STOCHASTIC_ROUND = getenv("STOCHASTIC_ROUND", 0)
MASTER_WEIGHTS = getenv("MASTER_WEIGHTS", 0)
ZERO_OPTIM = getenv("ZERO_OPTIM", 0)
FP8_AMAX_MARGIN = getenv("FP8_AMAX_MARGIN", 1.1)
IMMEDIATE_SCALE = getenv("IMMEDIATE_SCALE", 0)
MXFP8 = getenv("MXFP8", 0)
PRESTORE_WT = getenv("PRESTORE_WT", 0)  # pre-store transposed mxfp8 weight (W^T) so the dgrad skips dequant+transpose+requant
FUSED_ADAM_MXFP8 = getenv("FUSED_ADAM_MXFP8", 0)

def _gptoss_owned_shards(t:Tensor) -> tuple[Tensor, ...]:
  """Broadcast native Adam's owned shards without a local staging copy."""
  assert isinstance(t.device, tuple) and t.uop.axis == 0 and t.uop.has_buffer_identity(after_ok=True)
  node, deps = t.uop, []
  while node.op in (Ops.AFTER, Ops.RESHAPE, Ops.UNSHARD):
    if node.op is Ops.AFTER: deps.extend(node.src[1:])
    node = node.src[0]
  assert node is t.uop.storage_base
  # Preserve both FC1 writers (main and tail), exposing only their physical local
  # output view. Logical shard slicing would introduce a copy before each send.
  local = node.reshape(t.uop.shard_shape).after(*deps)
  return tuple(Tensor(local.mselect(i).copy_to_device(t.device)) for i in range(len(t.device)))

def _gptoss_gather_owned(t:Tensor) -> Tensor:
  return Tensor.cat(*_gptoss_owned_shards(t), dim=0)

def stochastic_round_bf16(x:Tensor) -> Tensor:
  bits = x.bitcast(dtypes.uint32)
  if isinstance(x.device, tuple):
    shape = x.uop.shard_shape if x.uop.axis is not None else x.shape
    noise = Tensor(UOp(Ops.MSTACK, src=tuple(Tensor.rand(*shape, device=d).uop for d in x.device)))
  else:
    noise = x.rand_like()
  noise = (noise * 0xFFFF).cast(dtypes.uint32)
  return ((bits + noise) & 0xFFFF0000).bitcast(dtypes.float32).cast(dtypes.bfloat16)

def clip_grads(grads:list[Tensor], grad_acc, clip_norm) -> Tensor:
  for g in grads: g.assign(g / grad_acc)
  total_norm = Tensor.stack(*[g.float().square().sum() for g in grads]).sum().sqrt().contiguous()
  for g in grads: g.assign((g * (clip_norm / (total_norm + 1e-6)).clamp(max_=1.0)).cast(g.dtype))
  return total_norm

def fclip_grads(grads:list[Tensor], clip_norm) -> tuple[list[Tensor], Tensor]:
  if getenv("FAST_GRAD_NORM", 0):
    from extra.gptoss_kernels.grad_norm import sum_squares_bf16
    squares = [sum_squares_bf16(g) if g.dtype == dtypes.bfloat16 else g.float().square().sum() for g in grads]
  else:
    squares = [g.float().square().sum() for g in grads]
  total_norm = Tensor.stack(*squares).sum().sqrt().contiguous()
  scale = (clip_norm / (total_norm + 1e-6)).clamp(max_=1.0)
  return [(g * scale).cast(g.dtype) for g in grads], total_norm

def clip_grads_lazy(grads:list[Tensor], grad_acc, clip_norm) -> tuple[list[Tensor], Tensor]:
  normalized = grads if grad_acc == 1 else [g / grad_acc for g in grads]
  if getenv("FAST_GRAD_NORM", 0):
    from extra.gptoss_kernels.grad_norm import sum_squares_bf16
    # The expert-weight gradients dominate this pass and are contiguous BF16 buffers. Keep uncommon dtypes and
    # small tensors on tinygrad's normal reduction path, where a custom launch would not repay its overhead.
    squares, local_squares = [], {}
    for g in normalized:
      if g.dtype == dtypes.bfloat16 and g.numel() >= 1_000_000:
        if getenv("GPTOSS_BATCHED_GRAD_NORM", 0) and isinstance(g.device, tuple) and g.uop.axis is not None:
          local_squares.setdefault(g.device, []).append(sum_squares_bf16(g, local=True))
        else: squares.append(sum_squares_bf16(g))
      else: squares.append(g.float().square().sum())
    # Sum each GPU's owned contributions before crossing devices. Replicated gradients above count only once.
    squares.extend(Tensor.stack(*parts, dim=1).sum(1).sum() for parts in local_squares.values())
  else:
    squares = [g.float().square().sum() for g in normalized]
  total_norm = Tensor.stack(*squares).sum().sqrt().contiguous()
  scale = (clip_norm / (total_norm + 1e-6)).clamp(max_=1.0)
  clipped = [(g * scale).cast(g.dtype) for g in normalized]
  # Same-shard fused optimizers can consume these directly and reproduce this BF16 rounding boundary in their kernel.
  for out, raw in zip(clipped, normalized):
    setattr(out, "_adam_raw_grad", raw)
    setattr(out, "_adam_clip_scale", scale)
  return clipped, total_norm

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

  def _zero_shard(self, t:Tensor) -> Tensor:
    if not self.zero or t.ndim < 2 or (t.shape[0] % len(self.device)) != 0: return t
    return Tensor(t.uop._shard(0, UOp.range(len(self.device), -1, AxisType.DEVICE)).unshard(0)).clone()

  def _zero_gather(self, t:Tensor) -> Tensor:
    if (deferred := getattr(self, '_deferred_lmhead', None)) is not None and deferred.stage(t): return deferred.parameter
    if not isinstance(t.device, tuple) or t.uop.axis != 0: return t
    # Finish local update math before selecting individual lanes. Otherwise gather can duplicate an update
    # containing a DEVICE range into single-device calls, where that range no longer has a launch binding.
    t = t.contiguous()
    n, sz = len(t.device), t.shape[0] // len(t.device)
    return Tensor.cat(*[t[p*sz:(p+1)*sz] for p in range(n)], dim=0)

  def fschedule_step(self, grads:list[Tensor]) -> list[Tensor]:
    if FUSED_ADAM_MXFP8 and self.master_params is not None and MXFP8 and not PRESTORE_WT:
      return self._fschedule_fused_adam_mxfp8(grads)
    updates, extra = self._step([], grads)
    for i, tt in enumerate(self.params): tt.assign(self._apply_update(tt, updates[i], self.master_params[i] if self.master_params else None))
    return self._scheduled_outputs(extra)

  def _scheduled_outputs(self, extra:list[Tensor]) -> list[Tensor]:
    fp8_inv_scales = [tt._inv_scale for tt in self.params if hasattr(tt, '_inv_scale')]
    fp8_next_inv_scales = [tt._next_inv_scale for tt in self.params if hasattr(tt, '_next_inv_scale')]
    fp8_wT = [tt._wT_q for tt in self.params if hasattr(tt, '_wT_q')] + [tt._wT_e8 for tt in self.params if hasattr(tt, '_wT_e8')]
    fp8_fc1_si = [tt._fc1_packed_si for tt in self.params if hasattr(tt, '_fc1_packed_si')]
    outputs = extra + self.params + self.buffers + (self.master_params or []) + fp8_inv_scales + fp8_next_inv_scales + fp8_wT + fp8_fc1_si
    return deferred.scheduled_outputs(outputs) if (deferred := getattr(self, '_deferred_lmhead', None)) is not None else outputs

  def _fschedule_fused_adam_mxfp8(self, grads:list[Tensor]) -> list[Tensor]:
    from extra.llama_kernels.fused_adam_mxfp8 import fused_adam_mxfp8, fused_adam_bf16_vocab
    self.b1_t *= self.b1
    self.b2_t *= self.b2
    for i, (tt, g) in enumerate(zip(self.params, grads)):
      if g.device != self.m[i].device: g = g.to(self.m[i].device)
      master = self.master_params[i]  # type: ignore[index]
      raw_grad, raw_scale = getattr(g, "_adam_raw_grad", None), getattr(g, "_adam_clip_scale", None)
      can_fuse = tt.dtype in dtypes.fp8s and g.dtype == dtypes.bfloat16 and master.shape[-1] % 32 == 0
      # The two 128256x2880 BF16 vocabulary matrices otherwise run separate m, v, and master kernels, with the
      # master kernel redundantly recomputing both moments. This is deliberately exact GPT-OSS-only dispatch.
      can_fuse_vocab = bool(getenv("GPTOSS_ADAM_BF16_VOCAB", 0)) and tt.dtype == dtypes.bfloat16 and \
        tt.shape == (128256, 2880)
      if can_fuse_vocab:
        raw_vocab_clip = bool(getenv("GPTOSS_ADAM_BF16_RAW_CLIP", 0)) and isinstance(raw_grad, Tensor) and \
          isinstance(raw_scale, Tensor) and raw_grad.dtype == dtypes.bfloat16 and raw_grad.shape == master.shape and \
          raw_grad.device == master.device and raw_scale.dtype == dtypes.float32 and raw_scale.shape in ((), (1,)) and \
          raw_scale.device == master.device and (not isinstance(master.device, tuple) or raw_grad.uop.axis == master.uop.axis or \
          (self.zero and raw_grad.uop.axis is None and master.uop.axis == 0))
        if raw_vocab_clip:
          # Vocab gradients are materialized after the DP backward reduction but remain replicated; select the same
          # ZeRO shard as the already-clipped path before applying its BF16 clip boundary inside the custom kernel.
          g = self._zero_shard(raw_grad) if isinstance(master.device, tuple) and raw_grad.uop.axis != master.uop.axis else raw_grad
        elif isinstance(g.device, tuple) and g.uop.axis != master.uop.axis:
          assert g.uop.axis is None and master.uop.axis == 0
          g = self._zero_shard(g)
        else:
          g = g.shard_like(master)
        self.m[i], self.v[i], self.master_params[i] = fused_adam_bf16_vocab(
          self.m[i], self.v[i], master, g, self.lr, self.b1_t, self.b2_t,
          b1=self.b1, b2=self.b2, eps=self.eps, clip_scale=raw_scale if raw_vocab_clip else None)
        # Replicas consume BF16 weights, not the FP32 master. Round each local shard before transferring it.
        new_w = self.master_params[i].cast(tt.dtype)  # type: ignore[index]
        tt.assign(self._zero_gather(new_w) if self.zero else new_w)
      elif can_fuse:
        has_fc1_si = hasattr(tt, '_fc1_packed_si')
        gptoss_down_shape = tt.ndim == 3 and tt.shape[-2:] == (3072, 3072) and not has_fc1_si
        raw_clip_target = (has_fc1_si and bool(getenv("GPTOSS_ADAM_RAW_CLIP", 0))) or \
          (gptoss_down_shape and bool(getenv("GPTOSS_ADAM_DOWN_RAW_CLIP", 0)))
        raw_clip = raw_clip_target and \
          isinstance(raw_grad, Tensor) and isinstance(raw_scale, Tensor) and \
          raw_grad.dtype == dtypes.bfloat16 and raw_grad.shape == master.shape and raw_grad.device == master.device and \
          raw_scale.dtype == dtypes.float32 and raw_scale.shape in ((), (1,)) and raw_scale.device == master.device and \
          (not isinstance(master.device, tuple) or raw_grad.uop.axis == master.uop.axis)
        if raw_clip:
          g = raw_grad
        elif isinstance(g.device, tuple) and g.uop.axis != master.uop.axis:
          assert g.uop.axis is None and master.uop.axis == 0
          g = self._zero_shard(g)
        else:
          g = g.shard_like(master)
        # Local optimizer state can make the fused kernel write SI straight into its persistent model buffer. ZeRO
        # needs the ordinary sharded temporary followed by the same gather used for q/e8.
        direct_si = tt._fc1_packed_si if has_fc1_si and not self.zero and master.device == tt._fc1_packed_si.device else None
        compact_q = self.zero and bool(getenv("GPTOSS_COMPACT_Q_GATHER", 0)) and (has_fc1_si or gptoss_down_shape)
        fused_out = fused_adam_mxfp8(self.m[i], self.v[i], master, g, self.lr, self.b1_t, self.b2_t,
                                     b1=self.b1, b2=self.b2, eps=self.eps,
                                     weight_decay=self.wd if tt.ndim >= 3 else 0.0,
                                     experts=tt.shape[0] if has_fc1_si else 1, out_si=direct_si,
                                     clip_scale=raw_scale if raw_clip else None, compact_q=compact_q)
        m, v, master, q, e8, *si_out = fused_out
        self.m[i], self.v[i], self.master_params[i] = m, v, master  # type: ignore[index]
        if self.zero:
          gather = _gptoss_gather_owned if compact_q and getenv("GPTOSS_OWNED_EXPERT_GATHER", 0) else self._zero_gather
          if compact_q:
            # Word-typed raw storage retains vectorized copies; restore zero padding in the gather assembly.
            assert q.dtype == dtypes.uint32
            if getenv("GPTOSS_COMPACT_GATHER_VECTOR", 0):
              from extra.llama_kernels.fused_adam_mxfp8 import gather_compact_q
              q = gather_compact_q(_gptoss_owned_shards(q), tt)
            else:
              q = gather(q).pad(
                ((0, 0), (0, tt.shape[1]-q.shape[1]), (0, tt.shape[2]//4-q.shape[2]))).bitcast(tt.dtype)
          else: q = gather(q)
          e8 = gather(e8)
          if si_out: si_out[0] = gather(si_out[0])
        if compact_q and getenv("GPTOSS_COMPACT_GATHER_VECTOR", 0): tt.replace(q)
        else: tt.assign(q.reshape(tt.shape))
        tt._inv_scale.assign(e8.reshape(tt._inv_scale.shape))
        if si_out:
          if direct_si is not None: tt._fc1_packed_si.replace(si_out[0])
          else: tt._fc1_packed_si.assign(si_out[0])
      else:
        m_new = self.b1 * self.m[i].float() + (1.0 - self.b1) * g.float()
        v_new = self.b2 * self.v[i].float() + (1.0 - self.b2) * (g.float() * g.float())
        self.m[i].assign(m_new.cast(self.m[i].dtype))
        self.v[i].assign(v_new.cast(self.v[i].dtype))
        update = self.lr * (m_new / (1.0 - self.b1_t)) / ((v_new / (1.0 - self.b2_t)).sqrt() + self.eps)
        tt.assign(self._apply_update(tt, update, master))
    return self._scheduled_outputs([self.b1_t, self.b2_t] + self.m + self.v)

  def fstep(self, grads:list[Tensor], grad_norm:Tensor|None=None):
    Tensor.realize(*([grad_norm] if grad_norm is not None else []), *self.fschedule_step(grads))

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
    wd = self.wd if t.ndim >= 3 else 0.0
    up = up.float().shard_like(w) + self.lr.to(w.device) * wd * w.detach()
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
        if hasattr(t, '_fc1_packed_si'):
          from extra.gemm.moe_gemm import mx_pack_3d
          packed_si = mx_pack_3d(new_e8)
          t._fc1_packed_si.assign(packed_si.shard_like(t._fc1_packed_si) if offloaded else packed_si)
        if PRESTORE_WT and hasattr(t, '_wT_q'):
          # pre-store W^T = quantize_mxfp8(dequant(stored w_q).transpose(1,2)) -- byte-exact with the dgrad's per-backward
          # recompute (uses ret==the stored w_q the dgrad reads, NOT new_w). Same op chain, done once here.
          from extra.gemm.cdna_asm_gemm import _mx_block_scale_3d
          w_phys = ret.cast(dtypes.bfloat16) * _mx_block_scale_3d(new_e8).cast(dtypes.bfloat16)
          wT_q, wT_e8, _ = quantize_mxfp8(w_phys.transpose(1, 2))
          t._wT_q.assign(wT_q.shard_like(t._wT_q) if offloaded else wT_q)
          t._wT_e8.assign(wT_e8.shard_like(t._wT_e8) if offloaded else wT_e8)
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
  def fstep(self, grads:list[Tensor], grad_norm:Tensor|None=None):
    offset = 0
    to_realize = []
    for o in self.optimizers:
      n = len(o.params)
      to_realize += o.fschedule_step(grads[offset:offset+n])
      offset += n
    Tensor.realize(*to_realize, *([grad_norm] if grad_norm is not None else []))
  @property
  def lr(self): return self.optimizers[0].lr
  @property
  def device(self): return self.optimizers[0].device
  @property
  def master_params(self):
    mp = [mp for o in self.optimizers for mp in (o.master_params or [])]
    return mp if mp else None
