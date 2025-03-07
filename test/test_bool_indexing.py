import unittest
import numpy as np
import torch
from tinygrad.tensor import Tensor

class TestBooleanIndexing(unittest.TestCase):
    def test_basic_boolean_indexing(self):
        # Create a tensor with random values and a matching boolean index for first dimension.
        x = Tensor.randn((5, 3, 7))
        x_np = x.numpy()
        bidx = Tensor([True, False, True, False, True])
        bidx_np = bidx.numpy()
        
        # Index using tinygrad and numpy.
        indexed_tensor = x[bidx]
        indexed_np = x_np[bidx_np]
        
        self.assertTrue(np.array_equal(indexed_tensor.numpy(), indexed_np))
    
    def test_incorrect_shape_boolean(self):
        # Boolean index with incompatible shape should raise an error.
        x = Tensor.randn((5, 3, 7))
        wrong_bidx = Tensor([True, False])  # Wrong shape: expecting length 5
        
        with self.assertRaises(Exception):
            _ = x[wrong_bidx]
    
    def test_all_false_indexing(self):
        # Test with boolean index returning an empty tensor.
        x = Tensor.randn((5, 3, 7))
        x_np = x.numpy()
        bidx = Tensor([False, False, False, False, False])
        bidx_np = bidx.numpy()
        
        indexed_tensor = x[bidx]
        indexed_np = x_np[bidx_np]
        
        # Expect both to be empty.
        self.assertEqual(indexed_tensor.numpy().size, 0)
        self.assertEqual(indexed_np.size, 0)
    
    def test_all_true_indexing(self):
        # Test with boolean index that selects all elements.
        x = Tensor.randn((5, 3, 7))
        x_np = x.numpy()
        bidx = Tensor([True]*5)
        bidx_np = bidx.numpy()
        
        indexed_tensor = x[bidx]
        indexed_np = x_np[bidx_np]
        
        self.assertTrue(np.array_equal(indexed_tensor.numpy(), indexed_np))
    
    def test_multi_dim_boolean_indexing(self):
        # Test a case of boolean indexing on a multidimensional tensor.
        # Here we perform boolean indexing on the last axis of a 2D tensor.
        a = Tensor.randn((4, 5))
        a_np = a.numpy()
        
        # Define a boolean index for the second axis
        bidx = Tensor([True, False, True, False, True])
        bidx_np = bidx.numpy()
        
        # First transpose so that boolean indexing acts on the first axis.
        a_transposed = a.transpose()
        a_transposed_np = a_np.transpose()
        
        indexed_tensor = a_transposed[bidx]
        indexed_np = a_transposed_np[bidx_np]
        
        self.assertTrue(np.array_equal(indexed_tensor.numpy(), indexed_np))

    def test_large_tensor_boolean_indexing(self):
        # Test with a large tensor using boolean indexing on the first dimension.
        x = Tensor.randn((20, 10, 20, 5))
        x_np = x.numpy()
        # Use torch as an additional reference.
        x_torch = torch.from_numpy(x_np)
        
        # Create a random boolean index for the first dimension.
        np.random.seed(42)
        bidx_np = np.random.choice(a=[True, False], size=(20,))
        bidx = Tensor(bidx_np.tolist())
        
        # Index using tinygrad, numpy, and torch.
        indexed_tensor = x[bidx]
        indexed_np = x_np[bidx_np]
        indexed_torch = x_torch[bidx_np].numpy()
        
        self.assertTrue(np.array_equal(indexed_tensor.numpy(), indexed_np))
        self.assertTrue(np.array_equal(indexed_tensor.numpy(), indexed_torch))
    
    def test_weird_tensor_boolean_indexing_torch(self):
        # Test boolean indexing on a high-dimensional tensor with an irregular pattern.
        a = Tensor.randn((13, 7, 5, 17))
        a_np = a.numpy()
        a_torch = torch.from_numpy(a_np)
        
        # Create a boolean index with a 'weird' pattern for the first dimension.
        pattern = [(i % 3 == 0) for i in range(13)]
        bidx = Tensor(pattern)
        bidx_np = np.array(pattern)
        
        # Index using tinygrad, numpy, and torch.
        indexed_tensor = a[bidx]
        indexed_np = a_np[bidx_np]
        indexed_torch = a_torch[bidx_np].numpy()
        
        self.assertTrue(np.array_equal(indexed_tensor.numpy(), indexed_np))
        self.assertTrue(np.array_equal(indexed_tensor.numpy(), indexed_torch))

if __name__ == '__main__':
    unittest.main()