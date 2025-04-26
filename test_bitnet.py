#!/usr/bin/env python3

from pathlib import Path
from extra.models.bitnet import build_transformer
import numpy as np

print("Loading BitNet model...")
model_path = Path.home() / '.cache/tinygrad/downloads/bitnet/model.safetensors'
model, raw_weights = build_transformer(model_path)

print("\nChecking if weights were loaded...")
# Check a few BitLinear weights
bitlinear_keys = [k for k in raw_weights.keys() if 'proj.weight' in k and not k.endswith('_scale')]
print(f"Found {len(bitlinear_keys)} BitLinear weight keys")

if bitlinear_keys:
    # Check the first BitLinear weight
    key = bitlinear_keys[0]
    weight = raw_weights[key]
    print(f"\nChecking {key}:")
    print(f"  Shape: {weight.shape}")
    print(f"  Dtype: {weight.dtype}")
    print(f"  Unique values: {np.unique(weight)[:10]}...")
    print(f"  Min/Max: {weight.min()}, {weight.max()}")
    
    # Check if it looks like packed ternary weights
    if weight.dtype == np.uint8:
        # Unpack a few values to check
        sample = weight.flat[:4]
        print(f"  First 4 packed bytes: {sample}")
        for i, byte in enumerate(sample):
            vals = []
            for j in range(4):
                val = (byte >> (2*j)) & 0b11
                vals.append(val - 1)  # Convert to ternary
            print(f"    Byte {i} ({byte:08b}) unpacks to: {vals}")

print("\nModel loaded successfully!") 