def test_fused_attention():
    from tinygrad.nn import fused_attention
    from tinygrad import Tensor
    import numpy as np

    q = Tensor.rand(1, 1, 4, 8)
    k = Tensor.rand(1, 1, 4, 8)
    v = Tensor.rand(1, 1, 4, 8)
    out = fused_attention(q, k, v)
    assert out.shape == (1, 1, 4, 8)
    np.testing.assert_allclose(out.numpy(), out.numpy(), atol=1e-6)  # Simple check