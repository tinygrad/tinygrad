def test_fused_attention():
    from tinygrad.nn import fused_attention
    from tinygrad import Tensor
    import numpy as np

    q = Tensor.rand(1, 1, 4, 8)
    k = Tensor.rand(1, 1, 4, 8)
    v = Tensor.rand(1, 1, 4, 8)
    out = fused_attention(q, k, v)
    assert out.shape == (1, 1, 4, 8)

    # Test with mask
    mask = Tensor.ones(1, 1, 4, 4)
    mask[:, :, 2:, :] = 0
    out_mask = fused_attention(q, k, v, mask)
    assert out_mask.shape == (1, 1, 4, 8)

    # Test scaling
    scale = q.shape[-1] ** -0.5
    sim = (q * scale) @ k.transpose(-2, -1)
    attn = sim.softmax(-1)
    expected = attn @ v
    np.testing.assert_allclose(out.numpy(), expected.numpy(), atol=1e-5)