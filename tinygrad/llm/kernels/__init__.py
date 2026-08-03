from typing import Any, cast
from tinygrad import Tensor, UOp, nn, dtypes
from tinygrad.llm.kernels.amd import Linear
from tinygrad.uop.ops import resolve

def select_cache_dtype(device:str|tuple[str, ...]|None, recurrent:bool, max_context:int):
  return dtypes.int8 if recurrent and max_context > 8192 and str(device).startswith("AMD") else dtypes.default_float

def make_attention_cache(batch:int|UOp, n_kv_heads:int, max_context:int, head_dim:int, device, recurrent:bool):
  dtype, cache_len = select_cache_dtype(device, recurrent, max_context), (max_context+255)//256*256 if recurrent else max_context
  shape = (2, batch, n_kv_heads, cache_len, head_dim)
  cache = Tensor.empty(*shape, dtype=dtype, device=device).contiguous()
  scale = Tensor.empty(*shape[:-1], dtype=dtypes.float16, device=device).contiguous() if dtype == dtypes.int8 else None
  return cache, scale, cache_len

def cached_attention(q:Tensor, stacked_kv:Tensor, cache_kv:Tensor, cache_scale:Tensor|None,
                     start_pos:int|UOp, max_context:int) -> Tensor:
  if cache_kv.dtype == dtypes.int8:
    from tinygrad.llm.kernels.amd import quantized_attention
    assert cache_scale is not None
    return quantized_attention(q, stacked_kv, cache_kv, cache_scale, start_pos, max_context)
  T = q.shape[2]
  assigned_kv = Tensor(cache_kv.uop.after(cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(stacked_kv.uop)))
  k, v = assigned_kv[0, :, :, 0:start_pos+T, :], assigned_kv[1, :, :, 0:start_pos+T, :]
  mask = Tensor.full((1, 1, T, k.shape[-2]), float("-inf"), dtype=q.dtype, device=q.device, buffer=False).triu(start_pos+1) \
    if resolve(T != 1) else None
  return q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor) -> Tensor:
  if str(q.device).startswith("AMD"):
    from tinygrad.llm.kernels.amd import gated_delta_prefill as kernel
  else:
    from tinygrad.llm.kernels.generic import gated_delta_prefill as kernel
  return kernel(q, k, v, beta, alpha, state)

def _prepare_quantized_weights(model:Any, state_dict:dict[str, Tensor]) -> list[tuple[Linear, Tensor]]:
  packed:list[tuple[Linear, Tensor]] = []
  layers = cast(dict[str, Linear], nn.state.get_state_dict(model, tensor_type=Linear))
  for name,owner in layers.items():
    key, weight = f"{name}.weight", state_dict[f"{name}.weight"]
    if str(weight.device).startswith("AMD") and (offset:=owner.set_quantized(weight)) is not None:
      packed.append((owner, offset))
      state_dict[key] = owner.weight
  return packed

def load_state_dict(model:Any, state_dict:dict[str, Tensor]):
  for key in nn.state.get_state_dict(model):
    if key.endswith(".ssm_beta_alpha.weight") and key not in state_dict:
      prefix = key.removesuffix("beta_alpha.weight")
      state_dict[key] = state_dict.pop(prefix+"beta.weight").cat(state_dict.pop(prefix+"alpha.weight"), dim=0).contiguous()
  packed = _prepare_quantized_weights(model, state_dict)
  nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)
  if packed: Tensor.realize(*(offset for _,offset in packed))
  for layer,offset in packed: layer._raw_offset_uop = offset.uop
