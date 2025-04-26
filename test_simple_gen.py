#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from extra.models.bitnet import BitNetConfig, BitNetForCausalLM, build_transformer
from examples.llama3 import Tokenizer

# Suppress debug output for cleaner results
import extra.models.bitnet as bitnet_module
bitnet_module.DEBUG_PRINT = False

print("Loading model...")
model_path = Path.home() / ".cache/tinygrad/downloads/bitnet/model.safetensors"
model, _ = build_transformer(model_path, load_weights=True)

print("Loading tokenizer...")
tokenizer_path = Path.home() / ".cache/tinygrad/downloads/bitnet/tokenizer.json"
tokenizer = Tokenizer(str(tokenizer_path))

# Test prompt
prompt = "The capital of France is"
print(f"\nPrompt: '{prompt}'")

# Tokenize
tokens = tokenizer.encode(prompt, bos=True)
print(f"Input tokens: {tokens}")

# Generate
print("\nGenerating...")
generated_tokens = tokens.copy()
past = None

for i in range(20):
    # Get next token
    input_tensor = Tensor([generated_tokens[-1]], dtype=dtypes.int32, device=Device.DEFAULT).unsqueeze(0)
    
    # Forward pass
    token, past, logits = model(input_tensor, past, temp=0.0)  # Greedy decoding
    
    # Get the actual logits for analysis
    logits_np = logits[0, -1, :].detach().to('CPU').numpy()
    
    # Find top 5 predictions
    top_indices = np.argsort(logits_np)[-5:][::-1]
    top_values = logits_np[top_indices]
    
    generated_tokens.append(token)
    token_str = tokenizer.decode([token])
    
    print(f"\nStep {i+1}:")
    print(f"  Generated token: {token} = '{token_str}'")
    print(f"  Top 5 predictions:")
    for idx, val in zip(top_indices, top_values):
        print(f"    {idx}: '{tokenizer.decode([int(idx)])}' (logit: {val:.3f})")
    
    # Check logits statistics
    print(f"  Logits stats: min={logits_np.min():.3f}, max={logits_np.max():.3f}, std={logits_np.std():.3f}")
    
    # Check if we're stuck
    if i > 3 and len(set(generated_tokens[-4:])) == 1:
        print("\n  WARNING: Stuck in repetitive loop!")
        break

# Decode full generation
generated_text = tokenizer.decode(generated_tokens)
print(f"\nFull generation: '{generated_text}'") 