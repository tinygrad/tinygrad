#!/usr/bin/env python3
"""Debug optimizer scope test

This script tests the LoRA implementation by:
1. Creating a small test Transformer model
2. Applying LoRA adapters to specific modules (wq, wv)
3. Freezing the base model parameters
4. Extracting only LoRA parameters for optimization
5. Creating an AdamW optimizer with the LoRA parameters

The test validates that the LoRA parameter extraction and optimizer
creation work correctly without scope or parameter access issues.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import AdamW
from examples.mlperf.llama2_70b_lora.lora import LoRAConfig, apply_lora_to_model, LoRAParameterManager
from extra.models.llama import Transformer

model_config = {
    "dim": 128,
    "hidden_dim": 256, 
    "n_heads": 4,
    "n_layers": 2,
    "norm_eps": 1e-5,
    "vocab_size": 1000,
    "max_context": 256,
    "jit": False
}
model = Transformer(**model_config)

print("=== BEFORE LORA ===")
all_params_before = get_parameters(model)
print(f"Total parameters: {len(all_params_before)}")

print(f"Layers: {len(model.layers)}")
for i, layer in enumerate(model.layers):
    print(f"Layer {i}:")
    print(f"  attention.wq.weight.shape: {layer.attention.wq.weight.shape}")
    print(f"  attention.wv.weight.shape: {layer.attention.wv.weight.shape}")

print("\n=== APPLYING LORA ===")
try:
    lora_config = LoRAConfig(r=4, alpha=8.0, target_modules=["wq", "wv"])
    original_modules = apply_lora_to_model(model=model, config=lora_config)
    print(f"LoRA applied successfully: {original_modules}")
    
    for i, layer in enumerate(model.layers):
        print(f"Layer {i} after LoRA:")
        if hasattr(layer.attention.wq, 'lora_A'):
            print(f"  wq.lora_A.weight.shape: {layer.attention.wq.lora_A.weight.shape}")
            print(f"  wq.lora_B.weight.shape: {layer.attention.wq.lora_B.weight.shape}")
        if hasattr(layer.attention.wv, 'lora_A'):
            print(f"  wv.lora_A.weight.shape: {layer.attention.wv.lora_A.weight.shape}")
            print(f"  wv.lora_B.weight.shape: {layer.attention.wv.lora_B.weight.shape}")
    
    print("\n=== FREEZING AND GETTING PARAMS ===")
    LoRAParameterManager.freeze_base_model(model=model)
    lora_params = LoRAParameterManager.get_lora_parameters(model=model)
    print(f"LoRA parameters: {len(lora_params)}")
    for i, param in enumerate(lora_params):
        print(f"  param[{i}].shape: {param.shape}")
    
    print("\n=== CREATING OPTIMIZER ===")
    optimizer = AdamW(lora_params, lr=1e-4)
    print("Optimizer created successfully!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()