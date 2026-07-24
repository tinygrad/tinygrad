"""Small, dependency-free Llama 2 LoRA semantics used by the MLPerf bring-up tests.

This is intentionally a toy model, not a Llama2-70B benchmark entry point.  It
keeps the two reference target names and the adapter checkpoint layout stable so
the full model can adopt the same training contract later.
"""

import math

from tinygrad import Tensor, nn
from tinygrad.nn.state import get_state_dict


LORA_TARGETS = ("qkv_proj", "o_proj")


class LoRALinear:
  """Frozen linear projection plus trainable ``B(A(dropout(x))) * alpha/rank``."""

  def __init__(self, in_features:int, out_features:int, rank:int=16, alpha:float=32.0, dropout:float=0.1, bias:bool=False):
    if rank <= 0: raise ValueError("rank must be positive")
    if not 0.0 <= dropout < 1.0: raise ValueError("dropout must be in [0, 1)")
    bound = 1 / math.sqrt(in_features)
    self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound).is_param_(False)
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound).is_param_(False) if bias else None
    # PEFT-compatible initialization: A is random and B is zero, making insertion an exact no-op.
    self.lora_A = Tensor.uniform(rank, in_features, low=-bound, high=bound)
    self.lora_B = Tensor.zeros(out_features, rank)
    self.rank, self.alpha, self.dropout = rank, alpha, dropout
    self.scaling = alpha / rank

  def base(self, x:Tensor) -> Tensor:
    return x.linear(self.weight.detach().transpose(), self.bias.detach() if self.bias is not None else None)

  def __call__(self, x:Tensor) -> Tensor:
    adapter = x.dropout(self.dropout).linear(self.lora_A.transpose()).linear(self.lora_B.transpose())
    return self.base(x) + adapter * self.scaling


class _FrozenLinear:
  def __init__(self, in_features:int, out_features:int, bias:bool=False):
    bound = 1 / math.sqrt(in_features)
    self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound).is_param_(False)
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound).is_param_(False) if bias else None

  def __call__(self, x:Tensor) -> Tensor:
    return x.linear(self.weight.detach().transpose(), self.bias.detach() if self.bias is not None else None)


def _frozen_rms_norm(x:Tensor, norm:nn.RMSNorm) -> Tensor:
  normalized = x * (x.square().mean(-1, keepdim=True) + norm.eps).rsqrt()
  return normalized * norm.weight.detach()


class _ToyLlama2Block:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, rank:int, alpha:float, dropout:float, norm_eps:float):
    if dim % n_heads: raise ValueError("dim must be divisible by n_heads")
    self.n_heads, self.head_dim = n_heads, dim // n_heads
    self.attention_norm = nn.RMSNorm(dim, norm_eps)
    self.attention_norm.weight.is_param_(False)
    self.qkv_proj = LoRALinear(dim, 3 * dim, rank, alpha, dropout, bias=False)
    self.o_proj = LoRALinear(dim, dim, rank, alpha, dropout, bias=False)
    self.ffn_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm.weight.is_param_(False)
    self.gate_proj = _FrozenLinear(dim, hidden_dim)
    self.up_proj = _FrozenLinear(dim, hidden_dim)
    self.down_proj = _FrozenLinear(hidden_dim, dim)

  def __call__(self, x:Tensor) -> Tensor:
    batch, seqlen, dim = x.shape
    qkv = self.qkv_proj(_frozen_rms_norm(x, self.attention_norm))
    q, k, v = qkv[..., :dim], qkv[..., dim:2*dim], qkv[..., 2*dim:]
    q = q.reshape(batch, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.reshape(batch, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.reshape(batch, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
    attention = q.scaled_dot_product_attention(k, v, is_causal=True).transpose(1, 2).reshape(batch, seqlen, dim)
    hidden = x + self.o_proj(attention)
    normed = _frozen_rms_norm(hidden, self.ffn_norm)
    return hidden + self.down_proj(self.gate_proj(normed).silu() * self.up_proj(normed))


class ToyLlama2LoRA:
  """Tiny causal transformer with LoRA only on fused QKV and output projections."""

  def __init__(self, vocab_size:int, dim:int, hidden_dim:int, n_heads:int, n_layers:int=1,
               rank:int=16, alpha:float=32.0, dropout:float=0.1, norm_eps:float=1e-5):
    if vocab_size <= 0 or dim <= 0 or hidden_dim <= 0 or n_layers <= 0: raise ValueError("model dimensions must be positive")
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.tok_embeddings.weight.is_param_(False)
    self.layers = [_ToyLlama2Block(dim, hidden_dim, n_heads, rank, alpha, dropout, norm_eps) for _ in range(n_layers)]
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.norm.weight.is_param_(False)
    self.output = _FrozenLinear(dim, vocab_size)

  def __call__(self, tokens:Tensor) -> Tensor:
    hidden = self.tok_embeddings.weight.detach()[tokens]
    for layer in self.layers: hidden = layer(hidden)
    return self.output(_frozen_rms_norm(hidden, self.norm))

  def loss(self, tokens:Tensor, labels:Tensor, ignore_index:int=-100) -> Tensor:
    return shifted_causal_loss(self(tokens), labels, ignore_index)


def shifted_causal_loss(logits:Tensor, labels:Tensor, ignore_index:int=-100) -> Tensor:
  """Mean next-token cross entropy after dropping the final logit and first label."""
  if logits.ndim != 3 or labels.ndim != 2: raise ValueError("expected logits [batch, sequence, vocab] and labels [batch, sequence]")
  if logits.shape[:2] != labels.shape or logits.shape[1] < 2: raise ValueError("logits and labels must have matching sequence shapes of length >= 2")
  shifted_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
  shifted_labels = labels[:, 1:].reshape(-1)
  losses = shifted_logits.sparse_categorical_crossentropy(shifted_labels, ignore_index=ignore_index, reduction="none")
  valid_count = shifted_labels.ne(ignore_index).sum().maximum(1)
  return losses.sum() / valid_count


def adapter_state_dict(model) -> dict[str, Tensor]:
  """Return adapters only, preserving deterministic model traversal names."""
  return {name:tensor for name, tensor in get_state_dict(model).items() if name.rsplit(".", 1)[-1] in ("lora_A", "lora_B")}


def adapter_parameters(model) -> list[Tensor]:
  return list(adapter_state_dict(model).values())


def backward_adapters(loss:Tensor, model) -> Tensor:
  """Populate gradients only for adapters (tinygrad's general ``backward`` visits every floating tensor)."""
  params = adapter_parameters(model)
  for param, grad in zip(params, loss.gradient(*params)):
    if param.grad is None: param.grad = grad
    else: param.grad.assign(param.grad + grad)
  return loss
