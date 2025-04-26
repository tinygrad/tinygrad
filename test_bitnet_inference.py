#!/usr/bin/env python3

from pathlib import Path
from extra.models.bitnet import build_transformer
import numpy as np
from tinygrad import Tensor

print("Loading BitNet model...")
model_path = Path.home() / '.cache/tinygrad/downloads/bitnet/model.safetensors'
model, raw_weights = build_transformer(model_path)

print("\nTesting BitLinear weight unpacking...")

# Get the first BitLinear layer
first_layer = model.model.layers[0].self_attn.q_proj

print(f"First BitLinear layer: q_proj")
print(f"  Weight shape: {first_layer.weight.shape}")
print(f"  Raw weight data available: {first_layer._raw_weight_data is not None}")

if first_layer._raw_weight_data is not None:
    print(f"  Raw weight data shape: {first_layer._raw_weight_data.shape}")
    print(f"  Raw weight data dtype: {first_layer._raw_weight_data.dtype}")
    print(f"  First few bytes: {first_layer._raw_weight_data.flat[:10]}")

# Test unpacking
print("\nTesting weight unpacking...")
try:
    unpacked = first_layer._get_unpacked_weights()
    print(f"  Unpacked weight shape: {unpacked.shape}")
    print(f"  Unpacked weight dtype: {unpacked.dtype}")
    
    # Get some values to check
    unpacked_np = unpacked.detach().to('CPU').numpy()
    unique_vals = np.unique(unpacked_np)
    print(f"  Unique values in unpacked weights: {unique_vals}")
    print(f"  Value counts: {[(val, (unpacked_np == val).sum()) for val in unique_vals]}")
    
except Exception as e:
    print(f"  ERROR unpacking weights: {e}")
    import traceback
    traceback.print_exc()

# Test a simple forward pass
print("\nTesting forward pass...")
test_input = Tensor.randn(1, 10, 2560)  # batch=1, seq=10, hidden=2560
try:
    output = first_layer(test_input)
    print(f"  Output shape: {output.shape}")
    
    # Check if output is all zeros
    output_np = output.detach().to('CPU').numpy()
    print(f"  Output stats - min: {output_np.min():.3f}, max: {output_np.max():.3f}, mean: {output_np.mean():.3f}")
    print(f"  Output all zeros: {np.all(output_np == 0)}")
    
except Exception as e:
    print(f"  ERROR in forward pass: {e}")
    import traceback
    traceback.print_exc() 