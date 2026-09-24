import pathlib
from dataclasses import replace
from typing import TYPE_CHECKING
from tinygrad.uop.ops import Ops
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import prod
from tinygrad.llm.gguf import gguf_read, ggml_data_to_tensor, GGUFTensor, _GGML_NATIVE, _GGML_QUANT
from tinygrad.llm.kernels.amd import HALFWORD_QUANTS, Q4_K, Q5_K, Q6_K, IQ4_XS, amd_custom_kernels_supported

if TYPE_CHECKING:
  from tinygrad.llm.model import TransformerConfig

def shard_config(config:"TransformerConfig", count:int) -> "TransformerConfig":
  assert config.ssm is not None
  assert all(d % count == 0 for d in (config.n_heads, config.n_kv_heads, config.hidden_dim, config.vocab_size,
    config.ssm.group_count, config.ssm.time_step_rank, config.ssm.inner_size)), 'uneven TP dimensions'
  return replace(config, n_heads=config.n_heads//count, n_kv_heads=config.n_kv_heads//count, hidden_dim=config.hidden_dim//count,
    ssm=replace(config.ssm, group_count=config.ssm.group_count//count, time_step_rank=config.ssm.time_step_rank//count,
                inner_size=config.ssm.inner_size//count))

def tp_sum(x:Tensor) -> Tensor:
  if not isinstance(x.device, tuple): return x
  padded = x.pad_to(x.max_shape)
  return Tensor(padded.uop.allreduce(Ops.ADD, x.device)).shrink(tuple((0, s) for s in x.shape))

def gguf_load_sharded(fn:Tensor|str|pathlib.Path, devices:tuple[str, ...]) -> tuple[dict, dict[str, Tensor]]:
  kv, state = gguf_read(fn)
  assert kv['general.architecture'] == 'qwen35' and not kv.get('qwen35.expert_count', 0), 'TP only supports dense qwen35 GGUF'
  if 'output.weight' not in state: state['output.weight'] = state['token_embd.weight']
  return kv, apply_shards(state, kv, devices)

def apply_shards(state:dict[str, GGUFTensor], kv:dict, devices:tuple[str, ...]) -> dict[str, Tensor]:
  assert len(devices) > 1 and len(set(devices)) == len(devices), "TP requires distinct devices"
  arch = kv['general.architecture']
  assert all(kv.get(f'{arch}.attention.{k}', len(devices)) % len(devices) == 0 for k in ('head_count', 'head_count_kv')), 'uneven heads'
  packed_kernels = all(amd_custom_kernels_supported(d) for d in devices)
  weights = {}
  for name,value in state.items():
    raw, shape, typ = value.data, value.shape, value.ggml_type
    if name == 'token_embd.weight':
      weights[name] = ggml_data_to_tensor(raw.to(devices[0]), prod(shape), typ).reshape(shape)
      continue
    key = name.split('.', 2)[-1] if name.startswith('blk.') else name
    axis = None  # Other weights are replicated.
    if key in ('attn_output.weight', 'ffn_down.weight', 'ssm_out.weight'): axis = 1
    elif key in ('attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_qkv.weight', 'attn_gate.weight', 'ssm_alpha.weight',
                 'ssm_beta.weight', 'ffn_gate.weight', 'ffn_up.weight', 'output.weight', 'ssm_conv1d.weight', 'ssm_a', 'ssm_dt.bias'): axis = 0
    if axis is not None: assert shape[axis] % len(devices) == 0, f"{name}: uneven TP split"
    local = tuple(s//len(devices) if i == axis else s for i,s in enumerate(shape))
    block, size = _GGML_QUANT[typ] if typ in _GGML_QUANT else (1, _GGML_NATIVE[typ].itemsize)
    storage = raw.reshape(*shape[:-1], shape[-1]//block, size)
    if axis == len(shape)-1:
      assert shape[-1] % (block*len(devices)) == 0, f'{name}: shard crosses a quantization block'
      storage = storage.to('CPU').realize()
    if axis is not None and (key.startswith('ssm_') or key in ('attn_qkv.weight', 'attn_gate.weight')):
      assert kv[f'{arch}.ssm.group_count'] % len(devices) == 0, 'uneven GDN heads'
      repeats = kv[f'{arch}.ssm.time_step_rank']//kv[f'{arch}.ssm.group_count']
      groups = ((kv[f'{arch}.ssm.group_count']*kv[f'{arch}.ssm.state_size'],)*2 + (kv[f'{arch}.ssm.inner_size']//repeats,)*repeats
                if key in ('attn_qkv.weight', 'ssm_conv1d.weight') else (shape[axis]//repeats,)*repeats)
      assert all(g % (len(devices)*(block if axis == len(shape)-1 else 1)) == 0 for g in groups), f'{name}: uneven TP group'
      parts = storage.to('CPU').realize().split(tuple(g//block if axis == len(shape)-1 else g for g in groups), dim=axis)
      pieces = [Tensor.cat(*(p.chunk(len(devices), dim=axis)[rank] for p in parts), dim=axis) for rank in range(len(devices))]
    else: pieces = storage.chunk(len(devices), dim=axis) if axis is not None else [storage]*len(devices)
    word = (dtypes.uint16 if typ in HALFWORD_QUANTS else dtypes.uint32) if packed_kernels and typ in (Q4_K, Q5_K, Q6_K, IQ4_XS) else dtypes.uint8
    shards = [p.contiguous().flatten().bitcast(word).to(d).clone().realize() for p,d in zip(pieces, devices)]
    data = Tensor(UOp.mstack(*(p.uop for p in shards)))
    data = ggml_data_to_tensor(data.bitcast(dtypes.uint8), prod(local), typ).reshape(local)
    weights[name] = data if packed_kernels and typ in (Q4_K, Q5_K, Q6_K, IQ4_XS) else data.realize()
  return weights
