def test_std():
    from tinygrad import Tensor
    import numpy as np

    x = Tensor([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(x.std().numpy(), 1.118033988749895, atol=1e-6)
    
    # Test dim
    x2d = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(x2d.std(0).numpy(), [1.0, 1.0], atol=1e-6)