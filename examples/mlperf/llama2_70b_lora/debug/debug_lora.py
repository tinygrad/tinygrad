#!/usr/bin/env python3
"""Debug LoRA application to transformer model"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from tinygrad.nn.state import get_parameters
from examples.mlperf.llama2_70b_lora.lora import LoRAConfig, apply_lora_to_model, LoRAParameterManager
from extra.models.llama import Transformer

# Create small test model
model = Transformer(
    dim=64, hidden_dim=128, n_heads=2, n_layers=1,
    norm_eps=1e-5, vocab_size=100, max_context=32, jit=False
)

print("=== BEFORE LORA ===")
print(f"Model layers: {len(model.layers)}")
print(f"First layer type: {type(model.layers[0])}")
print(f"Has attention: {hasattr(model.layers[0], 'attention')}")
print(f"Has feed_forward: {hasattr(model.layers[0], 'feed_forward')}")

if hasattr(model.layers[0], 'attention'):
    attention = model.layers[0].attention
    print(f"Attention attributes: {[attr for attr in dir(attention) if not attr.startswith('_')]}")
    print(f"Has wq: {hasattr(attention, 'wq')}")
    print(f"Has wv: {hasattr(attention, 'wv')}")

print(f"Total parameters before: {len(get_parameters(model))}")

# Apply LoRA
print("\n=== APPLYING LORA ===")
lora_config = LoRAConfig(r=4, alpha=8.0, target_modules=["wq", "wv"])
original_modules = apply_lora_to_model(model=model, config=lora_config)
print(f"Original modules replaced: {original_modules}")

print("\n=== AFTER LORA ===")
print(f"Total parameters after: {len(get_parameters(model))}")

# Check if LoRA was applied
if hasattr(model.layers[0], 'attention'):
    attention = model.layers[0].attention
    print(f"wq type: {type(attention.wq)}")
    print(f"wv type: {type(attention.wv)}")

# Try to collect LoRA parameters
lora_params = LoRAParameterManager.get_lora_parameters(model=model)
print(f"LoRA parameters found: {len(lora_params)}")

# Debug parameter collection
print("\n=== DEBUG PARAMETER COLLECTION ===")
def debug_collect_params(module, depth=0):
    indent = "  " * depth
    print(f"{indent}{type(module).__name__}")
    
    if hasattr(module, '__dict__'):
        for name, submodule in module.__dict__.items():
            if name.startswith('_'):
                continue
            print(f"{indent}  .{name}: {type(submodule).__name__}")
            if hasattr(submodule, '__dict__') and depth < 3:
                debug_collect_params(submodule, depth + 1)
    elif hasattr(module, '__iter__') and not isinstance(module, (str, bytes)):
        for i, submodule in enumerate(module):
            print(f"{indent}  [{i}]: {type(submodule).__name__}")
            if hasattr(submodule, '__dict__') and depth < 3:
                debug_collect_params(submodule, depth + 1)

debug_collect_params(model)