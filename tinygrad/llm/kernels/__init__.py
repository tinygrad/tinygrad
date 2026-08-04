from tinygrad import Tensor, UOp, nn, dtypes
from tinygrad.helpers import prod
from tinygrad.uop.ops import Ops, resolve

class Linear(nn.Linear):
  ggml_type:int|None = None
  def __init__(self, in_features:int, out_features:int, bias=True):
    super().__init__(in_features, out_features, bias)
    self.in_features, self.out_features = in_features, out_features
    self._raw_offset_uop:UOp|None = None
  def set_quantized(self, decoded:Tensor) -> Tensor|None:
    packed_sizes = {decoded.numel() // 256 * type_size:typ for typ,type_size in ((13, 176), (14, 210), (23, 136))}
    raw = next((u for u in decoded.uop.toposort() if u.op is Ops.SHRINK and u.dtype == dtypes.uint8 and prod(u.shape) in packed_sizes), None)
    if raw is None: return None
    self.weight, self.ggml_type = Tensor(raw).flatten(), packed_sizes[prod(raw.shape)]
    raw_offset = self.weight.uop.contiguous_view_offset()
    assert raw_offset is not None and raw_offset % 4 == 0 and self.weight.uop.buf_uop.dtype == dtypes.uint8
    if self.ggml_type == 23 and str(self.weight.device).startswith("AMD"):
      from tinygrad.llm.kernels.amd import iq4_half_lut
      iq4_half_lut(str(self.weight.device))
    return Tensor([raw_offset // 4], dtype=dtypes.uint64, device=self.weight.device)
  def __call__(self, x:Tensor) -> Tensor:
    if self.ggml_type in (13, 14, 23) and str(self.weight.device).startswith("AMD"):
      from tinygrad.llm.kernels.amd import q8_linear
      return q8_linear(self, x)
    return super().__call__(x)

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
  if str(q.device).startswith("AMD") and q.shape[-1] % 32 == 0 and v.shape[-1] % 4 == 0:
    from tinygrad.llm.kernels.amd import gated_delta_prefill as kernel
  else:
    from tinygrad.llm.kernels.generic import gated_delta_prefill as kernel
  return kernel(q, k, v, beta, alpha, state)
