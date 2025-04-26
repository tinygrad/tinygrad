#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from extra.models.bitnet import BitNetConfig, BitNetForCausalLM, build_transformer
from examples.llama3 import Tokenizer

# Test generation quality
print("Testing BitNet generation quality...")

# Load model
model_path = Path.home() / ".cache/tinygrad/downloads/bitnet/model.safetensors"
model, _ = build_transformer(model_path, load_weights=True)

# Load tokenizer
tokenizer_path = Path.home() / ".cache/tinygrad/downloads/bitnet/tokenizer.json"
tokenizer = Tokenizer(str(tokenizer_path))

# Test prompt
prompt = "The capital of France is"
print(f"\nPrompt: '{prompt}'")

# Tokenize
tokens = tokenizer.encode(prompt, bos=True)
print(f"Tokens: {tokens}")
print(f"Decoded tokens: {[tokenizer.decode([t]) for t in tokens]}")

# Convert to tensor
input_ids = Tensor(tokens, dtype=dtypes.int32, device=Device.DEFAULT).unsqueeze(0)
print(f"\nInput shape: {input_ids.shape}")

# Run through model to get logits
print("\n" + "="*50)
print("Running model forward pass...")

# Get initial logits
logits, _ = model(input_ids)
print(f"Logits shape: {logits.shape}")

# Analyze logits
logits_np = logits[0, -1, :].detach().to('CPU').numpy()
print(f"\nLogits statistics:")
print(f"  Min: {logits_np.min():.3f}")
print(f"  Max: {logits_np.max():.3f}")
print(f"  Mean: {logits_np.mean():.3f}")
print(f"  Std: {logits_np.std():.3f}")

# Check for NaN/Inf
if np.any(np.isnan(logits_np)):
    print("  WARNING: Logits contain NaN values!")
if np.any(np.isinf(logits_np)):
    print("  WARNING: Logits contain infinite values!")

# Get top predictions
top_k = 20
top_indices = np.argsort(logits_np)[-top_k:][::-1]
top_values = logits_np[top_indices]

print(f"\nTop {top_k} predictions:")
for i, (idx, val) in enumerate(zip(top_indices, top_values)):
    token_str = tokenizer.decode([int(idx)])
    print(f"  {i+1}. Token {idx}: '{token_str}' (logit: {val:.3f})")

# Check if all logits are similar (which would cause repetitive output)
unique_logits = np.unique(np.round(logits_np, 3))
print(f"\nNumber of unique logit values (rounded to 3 decimals): {len(unique_logits)}")
if len(unique_logits) < 100:
    print("  WARNING: Very few unique logit values, model may be collapsed!")

# Generate a few tokens
print("\n" + "="*50)
print("Generating tokens...")

generated_tokens = tokens.copy()
past = None

for i in range(20):
    # Get next token
    input_tensor = Tensor([generated_tokens[-1]], dtype=dtypes.int32, device=Device.DEFAULT).unsqueeze(0)
    
    # Forward pass with past
    token, past, _ = model(input_tensor, past, temp=0.0)  # Greedy decoding
    
    generated_tokens.append(token)
    token_str = tokenizer.decode([token])
    
    print(f"Step {i+1}: Generated token {token} = '{token_str}'")
    
    # Check if we're in a loop
    if i > 5 and len(set(generated_tokens[-5:])) == 1:
        print("  WARNING: Detected repetitive loop!")
        break

# Decode full generation
generated_text = tokenizer.decode(generated_tokens)
print(f"\nFull generation: '{generated_text}'")

# Test with different temperatures
print("\n" + "="*50)
print("Testing with different temperatures...")

for temp in [0.0, 0.5, 0.8, 1.0]:
    print(f"\nTemperature {temp}:")
    test_tokens = tokens.copy()
    past = None
    
    for i in range(10):
        input_tensor = Tensor([test_tokens[-1]], dtype=dtypes.int32, device=Device.DEFAULT).unsqueeze(0)
        token, past, _ = model(input_tensor, past, temp=temp)
        test_tokens.append(token)
    
    result = tokenizer.decode(test_tokens)
    print(f"  Result: '{result}'")

# Debug: Check layer outputs
print("\n" + "="*50)
print("Checking intermediate layer outputs...")

# Run a forward pass and capture intermediate outputs
x = model.model.embed_tokens(input_ids)
print(f"\nEmbedding output: shape={x.shape}, mean={x.numpy().mean():.3f}, std={x.numpy().std():.3f}")

# Check first few layers
for i in range(min(3, len(model.model.layers))):
    layer = model.model.layers[i]
    
    # Get position embeddings
    seq_len = x.shape[1]
    pos = Tensor.arange(seq_len, device=x.device)[None, :].expand(x.shape[0], -1)
    cos, sin = model.model.rotary(x, pos)
    
    # Run through layer
    x, _ = layer(x, cos, sin)
    
    x_np = x.detach().to('CPU').numpy()
    print(f"\nLayer {i} output: shape={x.shape}, mean={x_np.mean():.3f}, std={x_np.std():.3f}")
    
    # Check for issues
    if np.any(np.isnan(x_np)):
        print(f"  WARNING: Layer {i} output contains NaN!")
    if x_np.std() < 0.01:
        print(f"  WARNING: Layer {i} output has very low variance!")

print("\n" + "="*50)
print("Debugging complete.") 