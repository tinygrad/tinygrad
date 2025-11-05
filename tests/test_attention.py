def test_fused_attention():
    from tinygrad.nn import fused_attention
    from tinygrad import Tensor
    q = Tensor.rand(1, 1, 4, 8)
    k = Tensor.rand(1, 1, 4, 8)
    v = Tensor.rand(1, 1, 4, 8)
    out = fused_attention(q, k, v)
    assert out.shape == (1, 1, 4, 8)