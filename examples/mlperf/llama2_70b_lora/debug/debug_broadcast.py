#!/usr/bin/env python3
"""Debug broadcasting error"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from tinygrad import Tensor
from examples.mlperf.llama2_70b_lora.lora import LoRALinear

print("=== TESTING LORA FORWARD ===")

# Test case that should work
lora = LoRALinear(in_features=64, out_features=32, r=4, alpha=8.0)
x = Tensor.randn(2, 64)

print(f"Input shape: {x.shape}")
print(f"Linear weight shape: {lora.linear.weight.shape}")
print(f"LoRA A weight shape: {lora.lora_A.weight.shape}")
print(f"LoRA B weight shape: {lora.lora_B.weight.shape}")

try:
    output = lora(x)
    print(f"Output shape: {output.shape}")
    print("Forward pass successful")
except Exception as e:
    print(f"Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== TESTING MERGE OPERATIONS ===")
try:
    # Test merge
    lora.merge_weights()
    output_merged = lora(x)
    print(f"Output after merge shape: {output_merged.shape}")
    print("Merge successful")
    
    # Test unmerge 
    lora.unmerge_weights()
    output_unmerged = lora(x)
    print(f"Output after unmerge shape: {output_unmerged.shape}")
    print("Unmerge successful")
    
except Exception as e:
    print(f"Merge/unmerge failed: {e}")
    import traceback
    traceback.print_exc()