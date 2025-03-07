import unittest
import numpy as np
import torch
from tinygrad.tensor import Tensor

from .helpers import timeit

class TestBooleanIndexing(unittest.TestCase):
    def test_basic_boolean_indexing(self):
        x = Tensor.randn((5, 3, 7))
        x_np = x.numpy()
        bidx = Tensor([True, False, True, False, True])
        bidx_np = bidx.numpy()

        indexed_tensor, t_tiny = timeit(lambda: x[bidx])
        indexed_np, t_np = timeit(lambda: x_np[bidx_np])
        
        print(f"Basic indexing (tinygrad): {t_tiny:.6f}ms, (numpy): {t_np:.6f}ms")
        self.assertTrue(np.array_equal(indexed_tensor.numpy(), indexed_np))

    def test_incorrect_shape_boolean(self):
        x = Tensor.randn((5, 3, 7))
        wrong_bidx = Tensor([True, False])
        with self.assertRaises(Exception):
            _, _ = timeit(lambda: x[wrong_bidx])

    def test_all_false_indexing(self):
        x = Tensor.randn((5, 3, 7))
        x_np = x.numpy()
        bidx = Tensor([False] * 5)
        bidx_np = bidx.numpy()

        indexed_tensor, t_tiny = timeit(lambda: x[bidx])
        indexed_np, t_np = timeit(lambda: x_np[bidx_np])

        print(f"All-false indexing (tinygrad): {t_tiny:.6f}ms, (numpy): {t_np:.6f}ms")
        self.assertEqual(indexed_tensor.numpy().size, 0)
        self.assertEqual(indexed_np.size, 0)

    def test_all_true_indexing(self):
        x = Tensor.randn((5, 3, 7))
        x_np = x.numpy()
        bidx = Tensor([True] * 5)
        bidx_np = bidx.numpy()

        indexed_tensor, t_tiny = timeit(lambda: x[bidx])
        indexed_np, t_np = timeit(lambda: x_np[bidx_np])

        print(f"All-true indexing (tinygrad): {t_tiny:.6f}ms, (numpy): {t_np:.6f}ms")
        self.assertTrue(np.array_equal(indexed_tensor.numpy(), indexed_np))

    def test_multi_dim_boolean_indexing(self):
        a = Tensor.randn((4, 5))
        a_np = a.numpy()
        bidx = Tensor([True, False, True, False, True])
        bidx_np = bidx.numpy()

        # Transpose so that boolean indexing acts on the first axis.
        a_transposed = a.transpose()
        a_transposed_np = a_np.transpose()

        indexed_tensor, t_tiny = timeit(lambda: a_transposed[bidx])
        indexed_np, t_np = timeit(lambda: a_transposed_np[bidx_np])

        print(f"Multi-dim indexing (tinygrad): {t_tiny:.6f}ms, (numpy): {t_np:.6f}ms")
        self.assertTrue(np.array_equal(indexed_tensor.numpy(), indexed_np))

    def test_large_tensor_boolean_indexing(self):
        """
            Test the boolean indexing on a large tensor.

            Weirdly, in very large tensors, tinygrad (AMD) is faster than both numpy and torch.      
        """
        x = Tensor.randn((20, 100, 20, 50))
        x_np = x.numpy()
        x_torch = torch.from_numpy(x_np)
        
        np.random.seed(42)
        bidx_np = np.random.choice(a=[True, False], size=(20,))
        bidx = Tensor(bidx_np.tolist())
        
        indexed_tensor, t_tiny = timeit(lambda: x[bidx])
        indexed_np, t_np = timeit(lambda: x_np[bidx_np])
        indexed_torch, t_torch = timeit(lambda: x_torch[bidx_np].numpy())
        
        print(f"Large tensor indexing (tinygrad): {t_tiny:.6f}ms, (numpy): {t_np:.6f}ms, (torch): {t_torch:.6f}ms")
        self.assertTrue(np.array_equal(indexed_tensor.numpy(), indexed_np))
        self.assertTrue(np.array_equal(indexed_tensor.numpy(), indexed_torch))

    def test_weird_tensor_boolean_indexing_torch(self):
        a = Tensor.randn((13, 7, 5, 17))
        a_np = a.numpy()
        a_torch = torch.from_numpy(a_np)
        
        pattern = [(i % 3 == 0) for i in range(13)]
        bidx = Tensor(pattern)
        bidx_np = np.array(pattern)
        
        indexed_tensor, t_tiny = timeit(lambda: a[bidx])
        indexed_np, t_np = timeit(lambda: a_np[bidx_np])
        indexed_torch, t_torch = timeit(lambda: a_torch[bidx_np].numpy())
        
        print(f"Weird tensor indexing (tinygrad): {t_tiny:.6f}ms, (numpy): {t_np:.6f}ms, (torch): {t_torch:.6f}ms")
        self.assertTrue(np.array_equal(indexed_tensor.numpy(), indexed_np))
        self.assertTrue(np.array_equal(indexed_tensor.numpy(), indexed_torch))

if __name__ == '__main__':
    unittest.main()