#!/usr/bin/env python3
# Minimal reproduction case for Tinygrad gradient propagation issues (fixed)

# --- 0) Put env BEFORE tinygrad import ---
import os
# Choose ONE: CPU-only or METAL. Comment the other block.

# CPU-only
os.environ["CPU"] = "1"
os.environ["METAL"] = "0"

# # METAL-only
# os.environ["CPU"] = "0"
# os.environ["METAL"] = "1"

# --- 1) Now import tinygrad ---
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.nn import Embedding, Linear
from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import SGD
from tinygrad.engine.jit import TinyJit

# Force default float to float32 everywhere (important for METAL; good hygiene on CPU)
dtypes.default_float = dtypes.float32

print(f"Current device backend: {Device.DEFAULT}")

# Utility: check requires_grad of parameters
def debug_params(tag, params):
    print(f"{tag} parameter count: {len(params)}")
    for i, p in enumerate(params):
        print(f"  Param {i}: shape={p.shape}, dtype={p.dtype}, requires_grad={getattr(p, 'requires_grad', None)}")

# 1. Simple gradient test - works correctly
def test_scalar_grad():
    print("\n===== Test 1: Scalar to Scalar Gradient Propagation =====")
    x = Tensor([2.0], requires_grad=True, dtype=dtypes.float32)
    y = x * x
    loss = y.sum()
    print(f"x = {x.numpy()}, y = {y.numpy()}, loss = {loss.numpy()}")
    loss.backward()
    if hasattr(x, 'grad') and x.grad is not None:
        print(f"✓ Gradient calculation successful: dy/dx = {x.grad.numpy()}")
    else:
        print("✗ Gradient calculation failed: no gradient")

# 2. Embedding layer test
def test_embedding_grad():
    print("\n===== Test 2: Embedding Layer Gradient Propagation =====")
    embed = Embedding(10, 4)  # weights default to float32 with requires_grad=True
    x = Tensor([[1, 2, 3]], dtype=dtypes.int32)  # indices must be int

    params = list(get_parameters(embed))
    debug_params("Embedding", params)

    with Tensor.train():
        y = embed(x)                 # no .realize()
        print(f"Input shape: {x.shape}, Output shape: {y.shape}")
        loss = y.sum()               # no .realize()
        print(f"Loss: {loss.numpy()}")   # this will realize loss for printing
        loss.backward()

    grad_count = sum(1 for p in params if getattr(p, "grad", None) is not None)
    print(f"Parameters with gradients: {grad_count}/{len(params)}")
    for i, p in enumerate(params):
        has_grad = getattr(p, "grad", None) is not None
        print(f"  {'✓' if has_grad else '✗'} Param {i} ({'has' if has_grad else 'no'} gradient)")

# 3. Linear layer test
def test_linear_grad():
    print("\n===== Test 3: Linear Layer Gradient Propagation =====")
    linear = Linear(4, 8)
    x = Tensor(np.random.randn(2, 4).astype(np.float32))  # force float32

    params = list(get_parameters(linear))
    debug_params("Linear", params)

    with Tensor.train():
        y = linear(x)
        print(f"Input shape: {x.shape}, Output shape: {y.shape}")
        loss = y.sum()
        print(f"Loss: {loss.numpy()}")
        loss.backward()

    grad_count = sum(1 for p in params if getattr(p, "grad", None) is not None)
    print(f"Parameters with gradients: {grad_count}/{len(params)}")
    if grad_count == len(params):
        print("✓ Linear layer gradient propagation successful: all parameters have gradients")
    else:
        print("✗ Linear layer gradient propagation failed: some or all parameters have no gradients")

# 4. Simple training loop test using JIT
def test_jit_training():
    print("\n===== Test 4: Training Loop with JIT =====")
    class SimpleModel:
        def __init__(self):
            self.linear = Linear(4, 1)
        def __call__(self, x):
            return self.linear(x)

    model = SimpleModel()
    params = list(get_parameters(model))
    debug_params("JIT/Model", params)
    optimizer = SGD(params, lr=0.01)

    x = Tensor(np.random.randn(8, 4).astype(np.float32))
    y = Tensor(np.random.randn(8, 1).astype(np.float32))

    def train_step(x, y):
        pred = model(x)
        loss = ((pred - y)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    train_step_jit = TinyJit(train_step)

    try:
        with Tensor.train():
            loss = train_step_jit(x, y)
            print(f"Training loss: {loss.numpy()}")
            print("✓ JIT training step executed successfully")
    except Exception as e:
        print(f"✗ JIT training step failed: {type(e).__name__}: {e}")

# 5. Simple GPT model test
def test_simple_gpt():
    print("\n===== Test 5: Simplified GPT Model Gradient Propagation =====")
    class MiniGPT:
        def __init__(self, vocab_size=10, n_embd=16, max_len=5):
            self.token_embedding = Embedding(vocab_size, n_embd)
            self.pos_embedding = Embedding(max_len, n_embd)
            self.linear = Linear(n_embd, vocab_size)
        def __call__(self, idx, targets=None):
            token_emb = self.token_embedding(idx)   # (B, T, C)
            # positional indices as int32 (no grads needed for indices)
            pos = Tensor.arange(0, idx.shape[1], dtype=dtypes.int32).reshape(1, -1)
            pos_emb = self.pos_embedding(pos)       # (1, T, C)
            x = token_emb + pos_emb                 # broadcast add
            logits = self.linear(x)                 # (B, T, V)
            loss = None
            if targets is not None:
                logits_view = logits.reshape(-1, logits.shape[-1])
                targets_view = targets.reshape(-1)
                loss = logits_view.sparse_categorical_crossentropy(targets_view).mean()
            return logits, loss

    model = MiniGPT()
    params = list(get_parameters(model))
    debug_params("MiniGPT", params)

    x = Tensor([[1, 2, 3, 4]], dtype=dtypes.int32)
    y = Tensor([[2, 3, 4, 5]], dtype=dtypes.int32)

    with Tensor.train():
        logits, loss = model(x, y)
        print(f"Loss: {loss.numpy()}")
        loss.backward()

    grad_count = sum(1 for p in params if getattr(p, "grad", None) is not None)
    print(f"Parameters with gradients: {grad_count}/{len(params)}")
    comp = {"token_embedding": 0, "pos_embedding": 0, "linear": 0}
    # params ordering: token_embedding.weight, pos_embedding.weight, linear.weight, linear.bias
    for i, p in enumerate(params):
        has = getattr(p, "grad", None) is not None
        if i == 0: comp["token_embedding"] += int(has)
        elif i == 1: comp["pos_embedding"] += int(has)
        else: comp["linear"] += int(has)
    for k, v in comp.items():
        print(f"  {'✓' if v>0 else '✗'} {k}: {'has' if v>0 else 'no'} gradient")

# 6. Simulate overall training process
def test_training_process():
    print("\n===== Test 6: Complete Training Process Simulation =====")
    linear = Linear(4, 1)
    params = list(get_parameters(linear))
    optimizer = SGD(params, lr=0.01)
    debug_params("Training/Linear", params)

    x = Tensor(np.random.randn(8, 4).astype(np.float32))
    y = Tensor(np.random.randn(8, 1).astype(np.float32))

    with Tensor.train():
        pred = linear(x)
        loss = ((pred - y)**2).mean()
        print(f"Initial loss: {loss.numpy()}")

        optimizer.zero_grad()
        loss.backward()

        grad_count = sum(1 for p in params if getattr(p, "grad", None) is not None)
        print(f"Parameters with gradients: {grad_count}/{len(params)}")

        print("Attempting optimizer step...")
        optimizer.step()
        print("✓ Optimizer step successful")

        pred2 = linear(x)
        new_loss = ((pred2 - y)**2).mean()
        print(f"Loss after update: {new_loss.numpy()}")
        print(f"Loss change: {(loss.numpy() - new_loss.numpy()):.6f}")

def main():
    print("=== Tinygrad Gradient Propagation Issue Diagnosis (fixed) ===")
    print(f"Test environment: Tinygrad + {Device.DEFAULT}")

    test_scalar_grad()
    test_embedding_grad()
    test_linear_grad()
    test_jit_training()
    test_simple_gpt()
    test_training_process()

    print("\n=== Tests Complete ===")

def test_grad_propagation():
    main()
