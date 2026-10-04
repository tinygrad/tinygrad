"""GPT-OSS grouped updates and the persistent, deferred LM-head gather."""
import functools
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import compile_hip

# Advertised dense MXFP8 peak, not sparsity-doubled or adjusted for observed clocks.
MI350X_FP8_FLOPS = 4.614e15

def gptoss_model_flops(model, *, batch_size:int, seqlen:int) -> int:
  """Useful forward/backward GEMM FLOPs per node/update, using logical model dimensions.

  Excludes padding, inactive experts, recomputation, optimizer work, and small nonlinear/elementwise operations.
  Embedding lookup is not a GEMM; the vocabulary output projection is. A multiply-add counts as two FLOPs.
  """
  assert batch_size > 0 and seqlen > 0 and model.n_layers > 0
  assert 0 < model.experts_per_tok <= model.n_experts and model.sliding_window >= 0
  qkv_width = (model.n_heads + 2*model.n_kv_heads)*model.head_dim
  active_weights = model.n_layers * (model.dim*(qkv_width + model.n_heads*model.head_dim + model.n_experts) +
                                     3*model.experts_per_tok*model.dim*model.intermediate_size) + model.vocab_size*model.dim
  linear = 6*batch_size*seqlen*active_weights  # forward, dInput, dWeight
  # Layer zero is sliding; odd layer counts therefore have one more sliding layer.
  window = min(model.sliding_window or seqlen, seqlen)
  full_pairs = seqlen*(seqlen+1)//2
  window_pairs = window*seqlen-window*(window-1)//2
  pairs = (model.n_layers//2)*full_pairs + ((model.n_layers+1)//2)*window_pairs
  # QK and PV forward, plus their four derivative products. No QK recomputation.
  return linear + 12*batch_size*model.n_heads*model.head_dim*pairs

def gptoss_mfu(model_flops:int, *, updates:int, wall_s:float, device_count:int) -> float:
  """Useful-model percent MFU on wall time; independent of the executed-kernel GFLOPS counter."""
  assert model_flops > 0 and updates > 0 and wall_s > 0 and device_count > 0
  flops_per_second = model_flops*updates/wall_s
  return 100*flops_per_second/(device_count*MI350X_FP8_FLOPS)

def _has_allocated_storage(t):
  # is_realized follows .base, which does not unwrap an UNSHARD of a reshaped shard.
  # Check the actual owned storage and allocation, not a logical view predicate.
  return not t.uop.is_virtual and t.uop.has_buffer_identity(after_ok=True) and t.uop.storage_base.realized is not None

def initialize_persistent(optim, scheduler, grads, deferred=None):
  """Make all recurrence state real before recording: never replay zero/one initialization."""
  pending = list(grads)
  for optimizer in optim.optimizers:
    # In particular .realize() alone does NOT turn a virtual zero/one into storage.
    optimizer.m = [t.clone() for t in optimizer.m]
    optimizer.v = [t.clone() for t in optimizer.v]
    optimizer.b1_t = optimizer.b1_t.clone()
    optimizer.b2_t = optimizer.b2_t.clone()
    pending.extend([*optimizer.m,*optimizer.v,optimizer.b1_t,optimizer.b2_t,optimizer.lr])
  # Keep the actual scheduler/optimizer LR object shared, not a detached replacement.
  assert all(o.lr is optim.lr for o in optim.optimizers)
  scheduler.epoch_counter = scheduler.epoch_counter.clone()
  pending.append(scheduler.epoch_counter)
  if deferred is not None:
    assert not deferred.dirty and deferred.staged_updates == deferred.scheduled_updates == deferred.drains == 0
    pending.append(deferred.mailbox)
  for tensor in pending: tensor.realize()
  bad=[dict(index=i,shape=t.shape,op=t.uop.op.name,base=t.uop.storage_base.op.name,
    allocated=t.uop.storage_base.realized is not None,identity=t.uop.has_buffer_identity(after_ok=True))
    for i,t in enumerate(pending) if not _has_allocated_storage(t)]
  assert not bad, ('persistent capture state lacks owned storage',bad)
  unique = list({id(t):t for t in pending}.values())  # the only repeated object is the intentionally shared LR
  assert len({t.uop.storage_base for t in unique}) == len(unique), 'mutable state unexpectedly aliases'
  assert all(_has_allocated_storage(t) for t in optim.params+(optim.master_params or []))
  assert {t.uop.storage_base for t in unique}.isdisjoint(t.uop.storage_base for t in optim.params+(optim.master_params or []))
  return dict(tensors=len(pending),elements=sum(t.numel() for t in pending),
    logical_bytes=sum(t.numel()*t.dtype.itemsize for t in pending),unique_mutable_storage=len(unique),
    optimizer_lr_intentionally_shared=True,deferred_initialized=deferred is not None)

def next_group_size(*, step, sequences, batch_size, group_size, max_steps, eval_freq, checkpoint_freq=0, benchmark_steps=0):
  assert step >= 0 and sequences >= 0 and batch_size > 0 and group_size > 0
  assert max_steps >= 0 and eval_freq > 0 and checkpoint_freq >= 0 and benchmark_steps >= 0
  remaining = min(group_size, max_steps-step)
  if benchmark_steps: remaining = min(remaining, benchmark_steps-step)
  if remaining <= 0: return 0
  # Evaluation counts sequences; checkpoints and benchmark limits count optimizer updates.
  to_eval = ((sequences//eval_freq+1)*eval_freq-sequences+batch_size-1)//batch_size
  remaining = min(remaining, to_eval)
  if checkpoint_freq: remaining = min(remaining, checkpoint_freq-step%checkpoint_freq)
  return remaining

SOURCE='''
extern "C" __attribute__((device, const)) unsigned long __ockl_get_local_id(unsigned int);
extern "C" __attribute__((global)) void deferred_lmhead_forward_release(
    volatile unsigned short* mailbox, const volatile unsigned char* marker) {
  if (__ockl_get_local_id(0) == 0) {
    (void)marker[0];
    unsigned short bits = mailbox[0];
    mailbox[0] = bits;
  }
}
'''

@functools.cache
def release_kernel(mailbox:UOp, marker:UOp):
  assert mailbox.dtype == dtypes.bfloat16
  zero = UOp.const(0, dtypes.int32)
  # Only mailbox is written. The detached forward marker supplies an ordering edge, not a gradient.
  sink = UOp.sink(mailbox.base, marker.base, mailbox.index(zero).store(mailbox.index(zero).load()),
                  marker.index(zero).load(), UOp.special(64, 'lidx0'), UOp.special(1, 'gidx0'),
                  arg=KernelInfo('deferred_lmhead_forward_release', estimates=Estimates(mem=5)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                              UOp(Ops.SOURCE, arg=SOURCE), UOp(Ops.BINARY, arg=compile_hip(SOURCE, []))))

def release_grad(ctx, call): return ctx, None

def release_after(mailbox:Tensor, marker:Tensor):
  assert mailbox.device == marker.device
  return Tensor.custom_kernel(mailbox, marker.detach(), fxn=release_kernel, grad_fxn=release_grad)[0]

class DeferredLMHead:
  # Keep runtime coordination out of model/optimizer state_dict traversal. The canonical master and parameter
  # remain registered with the optimizer/model; the mailbox is drained before evaluation and checkpoints.
  __slots__ = ('parameter', 'optimizer', 'index', 'mailbox', 'gather', 'dirty', 'drains', 'staged_updates', 'scheduled_updates')

  def __init__(self, optimizer, parameter):
    from examples.mlperf import optim as implementation
    assert implementation.FUSED_ADAM_MXFP8 and implementation.MXFP8 and not implementation.PRESTORE_WT
    assert getenv('GPTOSS_ADAM_BF16_VOCAB', 0)
    opts = getattr(optimizer, 'optimizers', [optimizer])
    matches = [(o, i) for o in opts for i, p in enumerate(o.params) if p is parameter]
    assert len(matches) == 1
    opt, index = matches[0]
    assert opt.zero and opt.master_params is not None
    assert parameter.dtype == dtypes.bfloat16 and parameter.shape == (128256, 2880)
    assert parameter.uop.axis is None and opt.master_params[index].uop.axis == 0
    assert getattr(opt, '_deferred_lmhead', None) is None
    self.parameter, self.optimizer, self.index = parameter, opt, index
    self.mailbox = opt._zero_shard(parameter).realize()
    self.dirty, self.drains, self.staged_updates, self.scheduled_updates = False, 0, 0, 0
    self.gather = opt._zero_gather
    opt._deferred_lmhead = self

  def stage(self, value):
    # Identify the actual current master-derived BF16 value, not its shape or position in the schedule.
    if value.uop is not self.optimizer.master_params[self.index].cast(self.parameter.dtype).uop: return False
    self.mailbox.assign(value)
    self.staged_updates += 1
    return True

  def scheduled_outputs(self, outputs):
    assert self.staged_updates == self.scheduled_updates+1, 'Expected exactly one native LM-head shard update'
    self.scheduled_updates += 1
    return [p for p in outputs if p is not self.parameter]+[self.mailbox]

  def prefetch(self):
    self.parameter.assign(self.gather(self.mailbox))

  def prefetch_after(self, marker:Tensor):
    # Unconditional inside the captured forward: host dirty state must never control replay.
    self.parameter.assign(self.gather(release_after(self.mailbox, marker)))

  def mark_updated(self): self.dirty = True

  def drain(self):
    # Outside capture, leaving persistent buffer identities unchanged for subsequent replay.
    if not self.dirty: return
    self.prefetch()
    self.parameter.realize()
    self.dirty = False
    self.drains += 1
