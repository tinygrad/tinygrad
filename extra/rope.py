# which of these implement RoPE correctly?

from tinygrad import Tensor
import torch

def test_old_tinygrad(q, k, dim, end):
  def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
    freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
    return Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).reshape(1, end, 1, dim//2, 2)

  # (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
  def complex_mult(A, c, d):
    a,b = A[..., 0:1], A[..., 1:2]
    ro = a*c - b*d
    co = a*d + b*c
    return ro.cat(co, dim=-1)

  def apply_rotary_emb(xq:Tensor, xk:Tensor, freqs_cis:Tensor) -> tuple[Tensor, Tensor]:
    assert freqs_cis.shape[1] == xq.shape[1] == xk.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
    xq = xq.reshape(*xq.shape[0:-1], -1, 2)
    xk = xk.reshape(*xk.shape[0:-1], -1, 2)
    assert len(xq.shape) == len(xk.shape) == len(freqs_cis.shape) == 5
    c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
    xq_out = complex_mult(xq, c, d)
    xk_out = complex_mult(xk, c, d)
    return xq_out.flatten(3), xk_out.flatten(3)

  freqs_cis = precompute_freqs_cis(dim, end)
  return apply_rotary_emb(q, k, freqs_cis)

def test_new_tinygrad(q, k, dim, end):
  def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
    freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
    return Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).reshape(1, end, 1, dim//2, 2)

  # Copied from transformers.models.llama.modeling_llama.rotate_half
  def rotate_half(x): return Tensor.cat(-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2], dim=-1)
  def apply_rotary_emb(xq:Tensor, xk:Tensor, freqs_cis:Tensor) -> tuple[Tensor, Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = freqs_cis[..., 0:1].squeeze().repeat(1,1,1,2)
    sin = freqs_cis[..., 1:2].squeeze().repeat(1,1,1,2)
    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed, k_embed

  freqs_cis = precompute_freqs_cis(dim, end)
  return apply_rotary_emb(q, k, freqs_cis)

#https://github.com/meta-llama/llama/blob/main/llama/model.py#L80
def test_meta_llama(q, k, dim, end):
  def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

  def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

  def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

  freqs_cis = precompute_freqs_cis(dim, end)
  return apply_rotary_emb(q, k, freqs_cis)

def test_transformers_llama(q, k, dim, end):
  def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

  def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

  #from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

  base = 10000.0
  inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
  position_ids = torch.arange(end).unsqueeze(1)
  inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
  position_ids_expanded = position_ids[:, None, :].float()
  freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
  emb = torch.cat((freqs, freqs), dim=-1)
  cos = emb.cos()
  sin = emb.sin()
  return apply_rotary_pos_emb(q, k, cos, sin)

if __name__ == "__main__":
  shp = (1,2,1,4)
  q = Tensor([1.,2,3,4,5,6,7,8]).reshape(shp)
  k = Tensor([9.,10,11,12,13,14,15,16]).reshape(shp)
  qt = torch.Tensor([1.,2,3,4,5,6,7,8]).reshape(shp)
  kt = torch.Tensor([9.,10,11,12,13,14,15,16]).reshape(shp)
  q1, k1 = test_old_tinygrad(q, k, 4, 2)
  q2, k2 = test_new_tinygrad(q, k, 4, 2)
  q3, k3 = test_meta_llama(qt, kt, 4, 2)
  q4, k4 = test_transformers_llama(qt, kt, 4, 2)
  print("test_old_tinygrad", q1.numpy(), k1.numpy())
  print("test_new_tinygrad", q2.numpy(), k2.numpy())
  print("test_meta_llama", q3.numpy(), k3.numpy())
  print("test_transformers_llama", q4.numpy(), k4.numpy())
