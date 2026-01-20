from tinygrad.tensor import Tensor


def benchmark():
  BS, SEQ, DIM = 1, 4096, 64
  block_size = 64

  print(f"AETHER Sparsity Analysis: BS={BS}, SEQ={SEQ}, DIM={DIM}")

  q = Tensor.randn(BS, SEQ, DIM)
  k = Tensor.randn(BS, SEQ, DIM)

  # 1. Compute Geometry (Inline for benchmark)
  kb = k.reshape(BS, -1, block_size, DIM)
  centroids = kb.mean(axis=2)
  radii = ((kb - kb.mean(axis=2, keepdim=True)).square().sum(axis=3).sqrt().max(axis=2))

  # 2. Compute Upper Bound
  centroid_scores = (q.unsqueeze(2) * centroids.unsqueeze(1)).sum(axis=3)
  q_norm = (q**2).sum(axis=2).sqrt().unsqueeze(2)
  upper_bounds = centroid_scores + q_norm * radii.unsqueeze(1)

  # Analyze distribution to pick a fair threshold
  ub_data = upper_bounds.numpy().flatten()
  print(f"Upper Bound Stats: Min={ub_data.min():.2f}, Max={ub_data.max():.2f}, Mean={ub_data.mean():.2f}")

  import numpy as np
  adaptive_threshold = float(np.median(ub_data))

  # 3. Analyze Pruning
  mask = (upper_bounds > adaptive_threshold)

  total_blocks = mask.numel()
  kept_blocks = mask.sum().realize().item()
  pruned_blocks = total_blocks - kept_blocks
  sparsity = pruned_blocks / total_blocks * 100

  print(f"Adaptive Threshold (Median): {adaptive_threshold:.2f}")
  print(f"Total Blocks: {total_blocks}")
  print(f"Pruned Blocks: {pruned_blocks}")
  print(f"Sparsity: {sparsity:.2f}%")
  print("\n(This calculates theoretical FLOPs saved. 50% is expected with median threshold.)")

if __name__ == "__main__":
  benchmark()
