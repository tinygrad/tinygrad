import math
from typing import Union, Tuple, Optional

from tinygrad import Tensor, nn, dtypes
from tinygrad.helpers import prod, argfix, all_int

# rejection sampling truncated randn
def rand_truncn(*shape, dtype=None, truncstds=2, **kwargs) -> Tensor:
  CNT=8
  x = Tensor.randn(*(*shape, CNT), dtype=dtype, **kwargs)
  ctr = Tensor.arange(CNT).reshape((1,) * len(x.shape[:-1]) + (CNT,)).expand(x.shape)
  take = (x.abs() <= truncstds).where(ctr, CNT).min(axis=-1, keepdim=True)  # set to 0 if no good samples
  return (ctr == take).where(x, 0).sum(axis=-1)

# Use combined Linear congruential generator
def rand_lcg(*shape, device=None, dtype=None, **kwargs):
  if Tensor._rng_counter is None: Tensor._rng_counter = Tensor([0], dtype=dtypes.uint32, requires_grad=False)
  if (num := prod((shape:=argfix(*shape)))) == 0: return Tensor.zeros(shape, device=device, dtype=dtype, **kwargs)
  Tensor._rng_counter.assign(Tensor._rng_counter + num*2).realize()
  counts1 = Tensor.arange(num, device=device, dtype=dtypes.uint32, requires_grad=False)+Tensor._rng_counter.to(device) # Hack: Double arange to fuse
  counts2 = Tensor.arange(num, num*2, device=device, dtype=dtypes.uint32, requires_grad=False)+Tensor._rng_counter.to(device)
  counts1 = (counts1 * (2 ** 13) + Tensor._seed) ^ counts1
  counts2 = (counts2 * (2 ** 13) + Tensor._seed) ^ counts2

  m1, m2, a1, a2, b1, b2 = 4294967291, 2147483647, 1664525, 16807, 1013904223, 0
  counts1 = (x1 := a1 * counts1 + b1) - m1 * (x1.div(m1, upcast=False)).floor() # Cannot use %
  counts2 =  (x2 := a2 * counts2 + b2) - m2 * (x2.div(m2, upcast=False)).floor()
  out = (counts1 ^ counts2).cast(dtypes.float32).div(m1, upcast=False)
  out = out.reshape(shape).cast(dtypes.default_float if dtype is None else dtype)
  if (dtype or dtypes.default_float) == dtypes.half: out = out.clip(0, 1 - 0.001) # Avoid overflow
  out.requires_grad = kwargs.get("requires_grad")
  return out.contiguous()

# https://github.com/keras-team/keras/blob/v2.15.0/keras/initializers/initializers.py#L1026-L1065
def he_normal(*shape, a: float = 0.00, **kwargs) -> Tensor:
  std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:])) / 0.87962566103423978
  return std * rand_truncn(*shape, **kwargs)

class Conv2dHeNormal(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    self.in_channels, self.out_channels = in_channels, out_channels  # for testing
    self.weight = he_normal(out_channels, in_channels//groups, *self.kernel_size, a=0.0, dtype=dtypes.float32)
    if bias: self.bias = self.bias.cast(dtypes.float32)
  def __call__(self, x: Tensor):
    return x.conv2d(self.weight.cast(dtypes.default_float), self.bias.cast(dtypes.default_float) if self.bias is not None else None,
                    padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

class Linear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super().__init__(in_features, out_features, bias=bias)
    self.weight = Tensor.normal((out_features, in_features), mean=0.0, std=0.01, dtype=dtypes.float32)
    if bias: self.bias = Tensor.zeros(out_features, dtype=dtypes.float32)
  def __call__(self, x:Tensor):
    return x.linear(self.weight.cast(dtypes.default_float).transpose(), self.bias.cast(dtypes.default_float) if self.bias is not None else None)

class LinearBert(nn.Linear):
  def __init__(self, in_features, out_features, bias=True, std=0.02):
    self.weight = std * rand_truncn(out_features, in_features, dtype=dtypes.float32)
    self.bias = Tensor.zeros(out_features, dtype=dtypes.float32) if bias else None
  
  def __call__(self, x:Tensor):
    return x.linear(self.weight.cast(dtypes.default_float).transpose(), self.bias.cast(dtypes.default_float) if self.bias is not None else None)

class EmbeddingBert(nn.Embedding):
  def __init__(self, vocab_size:int, embed_size:int, std=0.02):
    self.vocab_sz, self.embed_sz = vocab_size, embed_size
    self.weight = std * rand_truncn(vocab_size, embed_size, dtype=dtypes.float32)

  def __call__(self, idx:Tensor) -> Tensor:
    if idx.numel() == 0: return Tensor.empty(idx.shape+(self.embed_sz,), dtype=self.weight.dtype, device=self.weight.device)
    arange_shp, weight_shp, big_shp = (1, 1, self.vocab_sz, 1), (1, 1, self.vocab_sz, self.embed_sz), idx.shape+(self.vocab_sz, self.embed_sz,)
    if not hasattr(self, 'arange'): self.arange = Tensor.arange(self.vocab_sz, requires_grad=False, device=self.weight.device).reshape(arange_shp)
    arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1,)).expand(big_shp), self.weight.cast(dtypes.default_float).reshape(weight_shp).expand(big_shp)
    return (arange == idx).mul(vals).sum(2, acc_dtype=vals.dtype)

class LayerNormBert:
  def __init__(self, normalized_shape:Union[int, Tuple[int, ...]], eps:float=1e-12, elementwise_affine:bool=True):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.weight, self.bias = (Tensor.ones(*self.normalized_shape, dtype=dtypes.float32), Tensor.zeros(*self.normalized_shape, dtype=dtypes.float32)) if elementwise_affine else (None, None)

  def __call__(self, x:Tensor):
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    xn = x.cast(dtypes.float32).layernorm(eps=self.eps, axis=self.axis).cast(x.dtype)
    if not self.elementwise_affine: return xn
    return (xn * self.weight.cast(dtypes.default_float) + self.bias.cast(dtypes.default_float))

def dropout_bert(x: Tensor, p=0.5):
  if not Tensor.training or p == 0: return x
  return x * (rand_lcg(*x.shape, requires_grad=False, device=x.device) >= p) * (1/(1.0 - p))

def scaled_dot_product_attention_bert(query:Tensor, key:Tensor, value:Tensor, attn_mask:Optional[Tensor]=None,
                                  dropout_p:float=0.0, is_causal:bool=False) -> Tensor:
  assert all_int(query.shape), f"does not support symbolic shape {query.shape}"
  if is_causal: attn_mask = Tensor.ones(query.shape[-2], key.shape[-2], requires_grad=False, device=query.device).tril(0).cast(dtypes.bool)
  if attn_mask is not None and attn_mask.dtype == dtypes.bool: attn_mask = (attn_mask == 0).where(-float("inf"), 0)
  qk = query @ key.transpose(-2,-1) / math.sqrt(query.shape[-1])
  return dropout_bert(((qk+attn_mask) if attn_mask is not None else qk).softmax(-1), dropout_p) @ value