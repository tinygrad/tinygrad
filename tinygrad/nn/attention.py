from tinygrad import Tensor
from typing import Optional

def fused_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None
) -> Tensor:
    """
    Fused scaled dot-product attention.
    
    Args:
        q: [batch, heads, seq_len, head_dim]
        k: [batch, heads, seq_len, head_dim]
        v: [batch, heads, seq_len, head_dim]
        mask: [batch, heads, seq_len, seq_len] or None
    
    Returns:
        [batch, heads, seq_len, head_dim]
    """
    # Scale
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