import unittest
from tinygrad.tensor import Tensor
from extra.aether import geometric_attention

class TestAether(unittest.TestCase):
  def test_geometric_attention_correctness(self):
    BS, SEQ, DIM = 2, 128, 32
    block_size = 32

    q = Tensor.rand(BS, SEQ, DIM)
    k = Tensor.rand(BS, SEQ, DIM)
    v = Tensor.rand(BS, SEQ, DIM)

    # 1. Standard Attention
    scores = q.dot(k.transpose(1, 2)) / (DIM ** 0.5)
    std_out = scores.softmax() @ v

    # 2. AETHER (Threshold=-inf to match dense)
    aether_out_full = geometric_attention(q, k, v, block_size=block_size, threshold=-float('inf'))

    # Check closeness
    diff = (std_out - aether_out_full).abs().max().realize()
    print(f"Max diff with threshold=-inf: {diff.numpy()}")
    self.assertLess(diff.numpy(), 1e-5)

if __name__ == '__main__':
  unittest.main()
