#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from extra.models.bitnet import BitNetConfig, BitNetForCausalLM, build_transformer, unpack_ternary_weights
from tinygrad.nn.state import safe_load

# Let's compare how HuggingFace unpacks weights vs our implementation

print("Loading weights to compare unpacking methods...")
model_path = Path.home() / ".cache/tinygrad/downloads/bitnet/model.safetensors"
weights = safe_load(str(model_path))

# Get a sample BitLinear weight
test_key = 'model.layers.0.self_attn.q_proj.weight'
test_weight = weights[test_key]
test_weight_np = test_weight.numpy()

print(f"\nOriginal packed weight:")
print(f"  Shape: {test_weight_np.shape}")
print(f"  Dtype: {test_weight_np.dtype}")
print(f"  Sample values: {test_weight_np.flatten()[:10]}")

# Our unpacking method
print("\n" + "="*50)
print("Our unpacking method:")
our_unpacked = unpack_ternary_weights(test_weight_np)
our_unpacked_np = our_unpacked.numpy()
print(f"  Shape: {our_unpacked_np.shape}")
print(f"  Unique values: {np.unique(our_unpacked_np)}")
print(f"  Distribution: -1: {np.sum(our_unpacked_np == -1)}, 0: {np.sum(our_unpacked_np == 0)}, 1: {np.sum(our_unpacked_np == 1)}")
print(f"  First few values: {our_unpacked_np.flatten()[:20]}")

# HuggingFace's unpacking method
print("\n" + "="*50)
print("HuggingFace unpacking method:")

# From the HF implementation:
# https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/blob/main/configuration_bitnet.py
VALUES_PER_ITEM = 4

def hf_unpack_weights(packed):
    """HuggingFace's exact unpack_weights implementation"""
    packed_shape = packed.shape
    
    if len(packed_shape) == 1:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim,)
    else:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim, *packed_shape[1:])

    unpacked = np.zeros(unpacked_shape, dtype=np.int8)

    # Unpack each row
    def unpack_single_row(row):
        unpacked_row = np.zeros(len(row) * VALUES_PER_ITEM, dtype=np.int8)
        for i in range(VALUES_PER_ITEM):
            unpacked_row[i::VALUES_PER_ITEM] = (row >> (2 * i)) & 0b11
        return unpacked_row

    if len(packed_shape) == 1:
        unpacked = unpack_single_row(packed)
    else:
        for i in range(packed_shape[0]):
            unpacked[i * VALUES_PER_ITEM:(i + 1) * VALUES_PER_ITEM] = unpack_single_row(packed[i]).reshape(VALUES_PER_ITEM, -1)

    # Convert to ternary
    unpacked = unpacked.astype(np.int8) - 1  # Map [0, 1, 2, 3] to [-1, 0, 1, 2] then clamp
    unpacked = np.clip(unpacked, -1, 1)  # Ensure only -1, 0, 1
    
    return unpacked.astype(np.float32)

hf_unpacked = hf_unpack_weights(test_weight_np)
print(f"  Shape: {hf_unpacked.shape}")
print(f"  Unique values: {np.unique(hf_unpacked)}")
print(f"  Distribution: -1: {np.sum(hf_unpacked == -1)}, 0: {np.sum(hf_unpacked == 0)}, 1: {np.sum(hf_unpacked == 1)}")
print(f"  First few values: {hf_unpacked.flatten()[:20]}")

# Compare the two methods
print("\n" + "="*50)
print("Comparison:")
if our_unpacked_np.shape == hf_unpacked.shape:
    diff = np.abs(our_unpacked_np - hf_unpacked)
    print(f"  Shapes match: {our_unpacked_np.shape}")
    print(f"  Max difference: {np.max(diff)}")
    print(f"  Are they equal? {np.allclose(our_unpacked_np, hf_unpacked)}")
    if not np.allclose(our_unpacked_np, hf_unpacked):
        print(f"  Number of different values: {np.sum(diff > 0)}")
        print(f"  First few differences:")
        diff_indices = np.where(diff > 0)[0][:10]
        for idx in diff_indices:
            print(f"    Index {idx}: ours={our_unpacked_np.flatten()[idx]}, HF={hf_unpacked.flatten()[idx]}")
else:
    print(f"  ERROR: Shapes don't match!")
    print(f"    Ours: {our_unpacked_np.shape}")
    print(f"    HF: {hf_unpacked.shape}")

# Check the actual BitLinear forward pass
print("\n" + "="*50)
print("Testing BitLinear forward pass...")

# Load model
model, _ = build_transformer(model_path, load_weights=True)
layer = model.model.layers[0].self_attn.q_proj

# Test input
test_input = Tensor.ones(1, 1, 2560, device=Device.DEFAULT)
print(f"Test input shape: {test_input.shape}")

# Get intermediate values
print("\nChecking activation quantization...")
x_int, x_scale = layer.activation_quant(test_input)
print(f"  Quantized input shape: {x_int.shape}")
print(f"  Input scale: {x_scale.numpy()}")

# Check weight
unpacked_w = layer._get_unpacked_weights()
print(f"\nUnpacked weight shape: {unpacked_w.shape}")
print(f"Weight scale: {layer.weight_scale.numpy()}")

# Matrix multiply
y = x_int @ unpacked_w.T
print(f"\nAfter matmul shape: {y.shape}")
y_np = y.numpy()
print(f"After matmul stats: min={y_np.min():.2f}, max={y_np.max():.2f}, mean={y_np.mean():.2f}")

# Post-process
result = layer.post_quant_process(y, x_scale, layer.weight_scale)
result_np = result.numpy()
print(f"\nFinal output shape: {result.shape}")
print(f"Final output stats: min={result_np.min():.2f}, max={result_np.max():.2f}, mean={result_np.mean():.2f}") 