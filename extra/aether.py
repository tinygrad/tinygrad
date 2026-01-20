from tinygrad.tensor import Tensor

from tinygrad.tensor import Tensor

def _pad_and_block(k, block_size):
  bs, seq, dim = k.shape
  if seq % block_size != 0:
    pad = block_size - (seq % block_size)
    k = k.pad2d((0, 0, 0, pad))
    return k, seq + pad
  return k, seq

def block_geometry(k: Tensor, block_size=64):
  k_padded, seq_padded = _pad_and_block(k, block_size)
  bs = k.shape[0]
  dim = k.shape[-1]
  k_blocks = k_padded.reshape(bs, seq_padded // block_size, block_size, dim)
  
  centroids = k_blocks.mean(axis=2)
  radii = ((k_blocks - centroids.unsqueeze(2)) ** 2).sum(axis=3).sqrt().max(axis=2)
  return centroids, radii

def geometric_attention(q:Tensor, k:Tensor, v:Tensor, block_size=64, threshold=0.1):
  # AETHER: Geometric Sparse Attention (O(N) approx)
  # Prunes blocks where max(q·k) < threshold using Cauchy-Schwarz upper bound
  
  bs, seq, dim = k.shape
  # 1. Compute Geometry (Lazy Eval)
  centroids, radii = block_geometry(k, block_size)

  # 2. Geometric Gate: max(q·k) <= q·μ + ||q||·r
  q_norm = (q**2).sum(axis=2).sqrt().unsqueeze(2)
  upper_bound = (q.unsqueeze(2) * centroids.unsqueeze(1)).sum(axis=3) + q_norm * radii.unsqueeze(1)
  
  # 3. Sparse Mask (1.0 = keep, 0.0 = prune)
  mask = (upper_bound > threshold).float()
  
  # Expansion logic needed for proper masking
  _, seq_padded = _pad_and_block(k, block_size) # Re-calc dimensions
  mask = mask.reshape(bs, seq, -1, 1).expand(bs, seq, seq_padded // block_size, block_size).reshape(bs, seq, seq_padded)
  
  # Crop mask back to original seq length if needed
  if seq_padded != seq:
    mask = mask[:, :, :seq]

  # 4. Masked Attention
  scores = q.dot(k.transpose(1, 2)) / (dim ** 0.5)
  scores = scores.masked_fill(mask == 0, -float('inf'))
  
  return scores.softmax() @ v
