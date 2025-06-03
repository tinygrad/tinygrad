#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear

def set_random_weights(layer, seed=1337):
    """Set random weights for a layer for testing purposes."""
    np.random.seed(seed)
    if hasattr(layer, 'weight'):
        if layer.weight.dtype == dtypes.uint8:
            # For packed weights, set random uint8 values
            layer.weight.assign(Tensor.randint(*layer.weight.shape, low=0, high=256, dtype=dtypes.uint8))
        else:
            layer.weight.assign(Tensor.randn(*layer.weight.shape))
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.assign(Tensor.randn(*layer.bias.shape))
    if hasattr(layer, 'weight_scale'):
        layer.weight_scale.assign(Tensor.ones(*layer.weight_scale.shape))

class TestBitLinear(unittest.TestCase):
    def test_bitlinear_creation(self):
        """Test that BitLinear can be created with different configurations."""
        from extra.models.bitnet import BitLinear
        
        # Test without bias
        layer = BitLinear(128, 64, bias=False)
        self.assertEqual(layer.in_features, 128)
        self.assertEqual(layer.out_features, 64)
        self.assertIsNone(layer.bias)
        self.assertEqual(layer.weight.shape, (16, 128))  # 64//4 = 16
        self.assertEqual(layer.weight.dtype, dtypes.uint8)
        
        # Test with bias
        layer_with_bias = BitLinear(128, 64, bias=True)
        self.assertIsNotNone(layer_with_bias.bias)
        self.assertEqual(layer_with_bias.bias.shape, (64,))
        self.assertEqual(layer_with_bias.bias.dtype, dtypes.float32)

    def test_weight_unpacking(self):
        """Test the weight unpacking functionality."""
        from extra.models.bitnet import BitLinear
        
        # Create test packed tensor
        packed = Tensor([[0b10100001, 0b00011000]], dtype=dtypes.uint8)
        unpacked = BitLinear.unpack_weights(packed, dtypes.float32)
        
        # Check unpacked shape
        expected_shape = (4, 2)  # 1*4, 2
        self.assertEqual(unpacked.shape, expected_shape)
        
        # The unpacked values should be in {-1, 0, 1}
        unpacked_np = unpacked.numpy()
        unique_values = np.unique(unpacked_np)
        self.assertTrue(all(val in [-1, 0, 1] for val in unique_values))

    def test_activation_quantization(self):
        """Test the activation quantization function."""
        from extra.models.bitnet import BitLinear
        
        layer = BitLinear(32, 16)
        
        # Create test input
        input_tensor = Tensor.randn(2, 32)
        
        # Test quantization
        quantized, scale = layer.activation_quant(input_tensor)
        
        # Check output properties
        self.assertEqual(quantized.dtype, dtypes.int8)
        self.assertEqual(quantized.shape, input_tensor.shape)
        self.assertEqual(scale.shape, (2, 1))  # Per-token scaling
        
        # Check quantization range
        quantized_np = quantized.numpy()
        self.assertTrue(np.all(quantized_np >= -128))
        self.assertTrue(np.all(quantized_np <= 127))

    def test_forward_pass(self):
        """Test that forward pass works and produces reasonable outputs."""
        from extra.models.bitnet import BitLinear
        
        # Set deterministic seed
        Tensor.manual_seed(1337)
        
        # Create layer
        layer = BitLinear(64, 32, bias=True)
        set_random_weights(layer)
        
        # Create test input
        batch_size, seq_len = 2, 8
        input_tensor = Tensor.randn(batch_size, seq_len, 64)
        
        # Forward pass
        output = layer(input_tensor)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, 32)
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(output.dtype, dtypes.float32)
        
        # Check that output is not all zeros
        output_np = output.numpy()
        self.assertFalse(np.allclose(output_np, 0))

    def test_forward_pass_without_bias(self):
        """Test forward pass without bias."""
        from extra.models.bitnet import BitLinear
        
        Tensor.manual_seed(1337)
        
        # Create layer without bias
        layer = BitLinear(64, 32, bias=False)
        set_random_weights(layer)
        
        # Create test input
        input_tensor = Tensor.randn(2, 64)
        
        # Forward pass
        output = layer(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 32))
        self.assertEqual(output.dtype, dtypes.float32)

    def test_caching_mechanism(self):
        """Test that weight caching works correctly."""
        from extra.models.bitnet import BitLinear
        
        layer = BitLinear(32, 16)
        set_random_weights(layer)
        
        # First call should create cache
        weights1 = layer._get_unpacked_weights()
        
        # Second call should use cache (same object)
        weights2 = layer._get_unpacked_weights()
        
        # Should be the same tensor (cached)
        self.assertTrue(weights1 is weights2)
        
        # Modifying weights should invalidate cache
        layer.weight.assign(Tensor.randint(*layer.weight.shape, low=0, high=256, dtype=dtypes.uint8))
        weights3 = layer._get_unpacked_weights()
        
        # Should be different now
        self.assertFalse(weights1 is weights3)

    def test_small_dimensions(self):
        """Test with small dimensions to ensure edge cases work."""
        from extra.models.bitnet import BitLinear
        
        # Test minimum viable dimensions
        layer = BitLinear(4, 4, bias=False)  # 4//4 = 1 for packed dimension
        set_random_weights(layer)
        
        input_tensor = Tensor.randn(1, 4)
        output = layer(input_tensor)
        
        self.assertEqual(output.shape, (1, 4))

    def test_consistency_across_calls(self):
        """Test that the same input produces the same output (deterministic)."""
        from extra.models.bitnet import BitLinear
        
        Tensor.manual_seed(42)
        
        layer = BitLinear(16, 8, bias=True)
        set_random_weights(layer, seed=42)
        
        # Create identical inputs
        input_tensor = Tensor.randn(1, 16)
        
        # Multiple forward passes should be identical
        output1 = layer(input_tensor)
        output2 = layer(input_tensor)
        
        np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-6)

    def test_different_batch_sizes(self):
        """Test that the layer works with different batch sizes."""
        from extra.models.bitnet import BitLinear
        
        layer = BitLinear(32, 16, bias=True)
        set_random_weights(layer)
        
        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            input_tensor = Tensor.randn(batch_size, 32)
            output = layer(input_tensor)
            self.assertEqual(output.shape, (batch_size, 16))

if __name__ == '__main__':
    unittest.main()
