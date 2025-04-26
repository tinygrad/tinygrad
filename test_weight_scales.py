#!/usr/bin/env python3

from pathlib import Path
from extra.models.bitnet import build_transformer
import numpy as np

print("Loading BitNet model...")
model_path = Path.home() / '.cache/tinygrad/downloads/bitnet/model.safetensors'
model, raw_weights = build_transformer(model_path)

print("\nChecking weight scales...")

# Check if weight scales are loaded
scale_keys = [k for k in raw_weights.keys() if k.endswith('.weight_scale')]
print(f"Found {len(scale_keys)} weight scale keys")

if scale_keys:
    # Check a few scales
    for i, key in enumerate(scale_keys[:5]):
        scale = raw_weights[key]
        print(f"\n{key}:")
        print(f"  Shape: {scale.shape}")
        print(f"  Value: {scale}")

# Check if scales are being used in BitLinear layers
print("\nChecking BitLinear layers...")
first_layer = model.model.layers[0].self_attn.q_proj

print(f"First BitLinear layer (q_proj):")
print(f"  Has weight_scale: {hasattr(first_layer, 'weight_scale')}")
if hasattr(first_layer, 'weight_scale'):
    print(f"  Weight scale value: {first_layer.weight_scale}")
    print(f"  Weight scale shape: {first_layer.weight_scale.shape}")
    print(f"  Weight scale dtype: {first_layer.weight_scale.dtype}")
else:
    print("  ERROR: No weight_scale attribute found!")

# Check the actual computation
print("\nChecking BitLinear forward pass...")
from tinygrad import Tensor

# Create a small test input
test_input = Tensor.randn(1, 1, 2560)
print(f"Test input shape: {test_input.shape}")
print(f"Test input stats: min={test_input.min().item():.3f}, max={test_input.max().item():.3f}, mean={test_input.mean().item():.3f}")

# Get unpacked weights
unpacked = first_layer._get_unpacked_weights()
print(f"\nUnpacked weights shape: {unpacked.shape}")
print(f"Unpacked weights unique values: {np.unique(unpacked.detach().to('CPU').numpy())}")

# Do the computation manually
print("\nManual computation:")
# Quantize input
x_norm = test_input.layernorm()
x_quant = x_norm.sign()
print(f"Quantized input shape: {x_quant.shape}")

# Matrix multiply
y = x_quant @ unpacked.T
print(f"After matmul shape: {y.shape}")
print(f"After matmul stats: min={y.min().item():.3f}, max={y.max().item():.3f}, mean={y.mean().item():.3f}")

# Apply scale
if hasattr(first_layer, 'weight_scale'):
    y_scaled = y * first_layer.weight_scale
    print(f"After scaling shape: {y_scaled.shape}")
    print(f"After scaling stats: min={y_scaled.min().item():.3f}, max={y_scaled.max().item():.3f}, mean={y_scaled.mean().item():.3f}")

# Now test the actual forward pass
print("\nActual forward pass:")
output = first_layer(test_input)
print(f"Output shape: {output.shape}")
output_np = output.detach().to('CPU').numpy()
print(f"Output stats: min={output_np.min():.3f}, max={output_np.max():.3f}, mean={output_np.mean():.3f}")
print(f"Output all zeros: {np.all(output_np == 0)}") 