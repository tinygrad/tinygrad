import unittest
from tinygrad import Tensor

class TestRangeify(unittest.TestCase):
  def test_expand_children(self):
    N = 1024
    A = Tensor.empty(N, N).sum(axis=1)
    ba = A.expand(N, N)
    ((ba+1).sum(axis=1) + (ba+2).sum(axis=0)).realize()

  def test_double_gemm(self):
    N = 1024
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    (A@B@C).realize()

  def test_double_gemm_exp(self):
    N = 1024
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    (((A@B).exp()@C).exp()).realize()

  def test_double_gemm_relu(self):
    N = 1024
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    (((A@B).relu()@C).relu()).realize()

  def test_double_gemm_relu_half_contig(self):
    N = 1024
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    (((A@B).relu().contiguous(arg=(1,))@C).relu()).realize()

  def test_double_gemm_half_contig(self):
    N = 1024
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    ((A@B).contiguous(arg=(1,))@C).realize()

  def test_double_gemm_contig(self):
    N = 1024
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    ((A@B).contiguous()@C).realize()

  def test_many_gemm(self):
    N = 1024
    A = Tensor.empty(N, N)
    B = Tensor.empty(N, N)
    C = Tensor.empty(N, N)
    D = Tensor.empty(N, N)
    E = Tensor.empty(N, N)
    F = Tensor.empty(N, N)
    (A@B@C@D@E@F).realize()

  def test_conv2d(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    x.conv2d(w1).realize()

  def test_conv2d_t(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    (x*2).conv2d(w1).realize()

  def test_double_conv2d(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    w2 = Tensor.empty(12, 8, 3, 3)
    x.conv2d(w1).conv2d(w2).realize()

  def test_double_conv2d_half_contig(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    w2 = Tensor.empty(12, 8, 3, 3)
    # NOTE: this contiguous doesn't help
    x.conv2d(w1).contiguous(arg=(1,)).conv2d(w2).permute(0,2,3,1).contiguous().realize()

  def test_double_conv2d_contig(self):
    x = Tensor.empty(1, 4, 32, 32)
    w1 = Tensor.empty(8, 4, 3, 3)
    w2 = Tensor.empty(12, 8, 3, 3)
    x.conv2d(w1).contiguous().conv2d(w2).realize()

  def test_transformer_ffn(self):
    from tinygrad.apps.llm import TransformerBlock
    from tinygrad import nn
    blk = TransformerBlock(1024, 4096, 1, 1, 1e-5)
    for p in nn.state.get_parameters(blk): p.replace(Tensor.empty(p.shape))

    x = Tensor.empty(128, 1024)
    out = blk._feed_forward(x)
    out.realize()

  def test_flash_attention(self):
    BS = 4
    HEADS = 2
    MATDIM = 16
    EMB = 8
    q = Tensor.empty(BS, HEADS, MATDIM, EMB)
    k = Tensor.empty(BS, HEADS, MATDIM, EMB)
    v = Tensor.empty(BS, HEADS, MATDIM, EMB)
    q.scaled_dot_product_attention(k, v).realize()

if __name__ == '__main__':
  unittest.main()
