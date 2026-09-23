from __future__ import annotations
from dataclasses import replace
from typing import TYPE_CHECKING, cast
from tinygrad import Tensor, nn
from tinygrad.helpers import get_child
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported
from tinygrad.uop.ops import Ops
if TYPE_CHECKING:
  from tinygrad.llm.model import Transformer, TransformerConfig

# Inside a block tensors have rank-local head/FFN shapes on a tuple of devices. This lets the existing custom kernels
# run independently on each GPU. Row-parallel output projections are summed before adding the replicated residual.
def sum_shards(x:Tensor) -> Tensor:
  if not isinstance(x.device, tuple): return x
  # Keep collectives statically sized even for symbolic prefill chunks.
  out = Tensor(x.pad_to(x.max_shape).contiguous().uop.allreduce(Ops.ADD, x.device))
  return out.shrink(tuple((0, s) for s in x.shape))

def shard_config(config:TransformerConfig, devices:tuple[str, ...]) -> TransformerConfig:
  n = len(devices)
  assert n > 1 and len(set(devices)) == n, "tensor parallelism requires distinct devices"
  assert not config.num_experts and not config.kv_lora_rank, "tensor parallelism currently supports dense MHA/GatedDeltaNet models"
  assert not config.ssm or not config.ssm.kda, "KDA tensor parallelism is not supported"
  assert all(v % n == 0 for v in (config.n_heads, config.n_kv_heads, config.hidden_dim, config.vocab_size)), "uneven tensor parallel split"
  ssm = config.ssm
  if ssm is not None:
    assert all(v % n == 0 for v in (ssm.group_count, ssm.time_step_rank, ssm.inner_size)), "uneven SSM head split"
    ssm = replace(ssm, group_count=ssm.group_count//n, time_step_rank=ssm.time_step_rank//n, inner_size=ssm.inner_size//n)
  return replace(config, n_heads=config.n_heads//n, n_kv_heads=config.n_kv_heads//n, hidden_dim=config.hidden_dim//n, ssm=ssm)

def _local_shard(t:Tensor, devices:tuple[str, ...], axis:int, splits:tuple[int, ...]=()) -> Tensor:
  # Q/K/V are packed segment-major in GGUF: shard each segment separately, then concatenate the rank-local segments.
  parts = t.split(splits, dim=axis) if splits else (t,)
  shards = [p.shard(devices, axis) for p in parts]
  assert all(p.uop.op is Ops.UNSHARD for p in shards)
  local = [Tensor(p.uop.src[0]) for p in shards]
  return local[0].cat(*local[1:], dim=axis).contiguous()

def load_sharded(model:Transformer, state_dict:dict[str, Tensor], config:TransformerConfig, devices:tuple[str, ...]):
  assert all(amd_custom_kernels_supported(d) for d in devices), "quantized tensor parallelism currently requires RDNA3/4"
  targets = nn.state.get_state_dict(model)
  quantized: dict[str, tuple[int, int, int]] = {}
  # Drop the large dequantization graphs before realizing any shards. Otherwise each realization traverses all those live Tensor graphs.
  for name in targets:
    if not name.endswith('.weight') or not isinstance(get_child(model, name.rsplit('.', 1)[0]), Linear): continue
    out_features, in_features = map(int, state_dict[name].shape)
    packed = Linear(in_features, out_features, bias=False)
    packed.set_quantized(state_dict[name])
    if packed.ggml_type is not None:
      quantized[name] = (packed.ggml_type, out_features, in_features)
      state_dict[name] = packed.weight

  for name, target in targets.items():
    weight = state_dict.pop(name)
    shape = quantized[name][1:] if name in quantized else weight.shape
    module = get_child(model, name.rsplit('.', 1)[0])
    axis: int|None = None
    splits: tuple[int, ...] = ()
    if isinstance(module, Linear):
      axis = 1 if name.endswith(('attn_output.weight', 'ffn_down.weight', 'ssm_out.weight')) else 0
      if name.endswith('.bias'): axis = 0
    if any(s in name for s in ('ssm_conv1d.', 'ssm_dt.')) or name.endswith('.ssm_a'): axis = 0
    if config.ssm is not None and name.startswith('blk.') and config.ssm_layers[int(name.split('.')[1])]:
      ssm = config.ssm
      repeats = ssm.time_step_rank // ssm.group_count
      # GatedDeltaNet repeats the entire K-head group, not each individual head. Keep V heads with their corresponding Q/K heads.
      if 'attn_qkv.' in name or 'ssm_conv1d.' in name:
        q_dim = ssm.group_count * ssm.state_size
        splits = (q_dim, q_dim) + (ssm.inner_size // repeats,) * repeats
      elif axis is not None:
        splits = (cast(int, shape[axis]) // repeats,) * repeats if any(s in name for s in
          ('attn_gate.', 'ssm_alpha.', 'ssm_beta.', 'ssm_out.', 'ssm_dt.')) or name.endswith('.ssm_a') else ()
    if name == 'token_embd.weight':
      # Lookup on GPU 0; only the selected embedding vectors are broadcast, not the whole vocabulary.
      target.replace(weight.to(devices[0]))
      continue
    if name in quantized:
      assert isinstance(module, Linear) and axis is not None
      typ, out_features, in_features = quantized[name]
      # Host shard copies may vectorize word loads; GGUF views are sometimes only halfword-aligned.
      # set_quantized's contiguous is a storage-preserving view, not a dequantization/materialization.
      if weight.uop.op is Ops.STAGE: weight = Tensor(weight.uop.src[0])
      if (offset:=weight.uop.contiguous_view_offset()) is not None and offset % 16:
        weight = weight.bitcast('uint8').clone().realize().bitcast(weight.dtype)
      raw = weight.reshape(out_features, in_features//256, -1)
      assert axis != 1 or raw.shape[1] % len(devices) == 0, "input shards must align to GGML super-blocks"
      if axis == 1:
        assert all(s % (256 * len(devices)) == 0 for s in splits), "SSM head groups must align to GGML super-blocks"
        splits = tuple(s // 256 for s in splits)
      module.ggml_type = typ
      module.weight = _local_shard(raw, devices, axis, splits).reshape(-1)
      continue
    sharded = weight.to(devices).contiguous() if axis is None else _local_shard(weight, devices, axis, splits)
    assert sharded.shape == target.shape, f"{name}: {sharded.shape} != {target.shape}"
    target.replace(sharded)
  # Realize packed shards together; leave the embedding lazy so only the selected rows are dequantized.
  Tensor.realize(*[p for name,p in nn.state.get_state_dict(model).items() if name != 'token_embd.weight'])
