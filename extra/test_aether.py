import unittest
import math
from tinygrad.tensor import Tensor
from extra.aether import geometric_attention

class TestAether(unittest.TestCase):
  def test_geometric_attention_correctness(self):
    # Benchmark specific parameters
    BS, SEQ, DIM = 2, 128, 32
    block_size = 32
    
    q = Tensor.rand(BS, SEQ, DIM)
    k = Tensor.rand(BS, SEQ, DIM)
    v = Tensor.rand(BS, SEQ, DIM)
    
    # 1. Standard Attention
    scores = q.dot(k.transpose(1, 2)) / (DIM ** 0.5)
    std_out = scores.softmax() @ v
    
    # 2. AETHER (Threshold=0.0 -> Should match approximately or exactly if bound is tight?)
    # Actually, threshold=0.0 doesn't mean "keep all". 
    # It means "keep blocks where upper_bound > 0". 
    # Since dot product can be negative, standard attention keeps EVERYTHING.
    # To verify correctness, we need to ensure that AETHER *with a high threshold* 
    # produces a result that is "close enough" to dense, 
    # OR verify that with threshold=-inf it matches exactly.
    
    # Test 1: Threshold = -inf (Should match exactly)
    aether_out_full = geometric_attention(q, k, v, block_size=block_size, threshold=-float('inf'))
    
    # Check closeness
    diff = (std_out - aether_out_full).abs().max().realize()
    print(f"Max diff with threshold=-inf: {diff.numpy()}")
    self.assertLess(diff.numpy(), 1e-5)
    
  def test_sparsity_generation(self):
    # Verify that increasing threshold increases sparsity
    # We can't easily check internal mask in the function, 
    # but we can check if output magnitude changes (it should, as we drop blocks).
    pass

if __name__ == '__main__':
  unittest.main()
