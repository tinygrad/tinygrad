import numpy as np
import torch
from tinygrad import Tensor
import os, sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extra.torch_backend.backend import index_tensor, unwrap, wrap

# Test 1: Simple advanced indexing
def test_index_tensor_simple():

     # Create a tinygrad tensor.
    x_tiny = Tensor([[10, 20, 30], [40, 50, 60]])
    # For dimension 0, use None to indicate "all", and for dimension 1, use advanced index.
    j_torch = torch.tensor([0, 2])
    # Wrap inputs.
    x_wrapped = wrap(x_tiny)
    j_wrapped = wrap(Tensor(j_torch.numpy(), device="CPU", dtype=Tensor(j_torch.numpy()).dtype))
    # Call index_tensor with a None for the first dimension.
    result = index_tensor(x_wrapped, [None, j_wrapped])
    result_np = unwrap(result).realize().numpy()

    # Expected: this should behave like x[:, j] in torch.
    x_torch = torch.tensor([[10, 20, 30], [40, 50, 60]])
    expected_np = x_torch.index([None, j_torch]).numpy()  # equivalent to x_torch[:, j_torch]
    np.testing.assert_equal(result_np, expected_np)

    # Create a tinygrad tensor.
    x_tiny = Tensor([[7, 8, 9], [10, 11, 12]])
    # Create tinygrad Tensors for indices.
    i_tiny = Tensor([1, 0], dtype=Tensor([1, 0]).dtype, device="CPU")
    j_tiny = Tensor([2, 1], dtype=Tensor([2, 1]).dtype, device="CPU")
    # Directly call index_tensor without additional wrapping.
    result = index_tensor(wrap(x_tiny), [wrap(i_tiny), wrap(j_tiny)])
    result_np = unwrap(result).realize().numpy()

    # Expected result using torch.
    x_torch = torch.tensor([[7, 8, 9], [10, 11, 12]])
    i_torch = torch.tensor([1, 0])
    j_torch = torch.tensor([2, 1])
    expected_np = x_torch.index([i_torch, j_torch]).numpy()
    np.testing.assert_equal(result_np, expected_np)

    # if __name__ == "__main__":
    #     test_index_tensor_simple()
    #     print("All index_tensor tests passed!")
test_index_tensor_simple()