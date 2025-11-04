from tinygrad import Tensor

def attention(q:Tensor, k:Tensor, v:Tensor, mask:Tensor|None=None) -> Tensor:
  """
  Scaled dot-product attention.
  q, k, v: [batch, heads, seq_len, head_dim]
  mask: [batch, seq_len, seq_len] or None
  """
  scale = q.shape[-1] ** -0.5
  q = q * scale
  
  # Q @ K^T
  sim = q @ k.transpose(-2, -1)  # [batch, heads, seq_len, seq_len]
  
  # Apply mask
  if mask is not None:
    sim = sim.masked_fill(mask == 0, -float('inf'))
  
  # Softmax over last dim
  attn = sim.softmax(-1)
  
  # Attn @ V
  return attn @ v  # [batch, heads, seq_len, head_dim]