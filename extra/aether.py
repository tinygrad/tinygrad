from tinygrad.tensor import Tensor

def geometric_attention(q:Tensor, k:Tensor, v:Tensor, block_size=64, threshold=0.1):
  # 1. Geometry: Reshape (B, NB, BS, D) -> Centroids (B, NB, D), Radii (B, NB)
  B, S, D = q.shape
  kb = k.reshape(B, -1, block_size, D)
  c, r = kb.mean(axis=2), ((kb - kb.mean(axis=2, keepdim=True)).square().sum(axis=3).sqrt().max(axis=2))

  # 2. Pruning: Upper Bound Check (Cauchy-Schwarz)
  # q: (B, S, 1, D) * c: (B, 1, NB, D) -> (B, S, NB)
  bound = (q.unsqueeze(2) * c.unsqueeze(1)).sum(axis=3) + q.square().sum(axis=2).sqrt().unsqueeze(2) * r.unsqueeze(1)

  # 3. Masked Attention (Fused)
  mask = (bound > threshold).reshape(B, S, -1, 1).expand(B, S, S//block_size, block_size).reshape(B, S, S)
  return q.dot(k.transpose(1, 2)).div(D**0.5).masked_fill(~mask, -float('inf')).softmax().dot(v)
