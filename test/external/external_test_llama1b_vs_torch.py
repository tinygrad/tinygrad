#!/usr/bin/env python3
"""
Llama 1B Speed Test: TinyGrad vs PyTorch on CPU

This test validates that tinygrad is faster than PyTorch for Llama 1B inference on CPU.
Bounty requirement: "llama 1B faster than torch on CPU in CI (no weight download needed, just model speed. either LLVM or CLANG okay)"

Usage:
  python3 test/external/external_test_llama1b_vs_torch.py
  LLVM=1 python3 test/external/external_test_llama1b_vs_torch.py
  CLANG=1 python3 test/external/external_test_llama1b_vs_torch.py
"""
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import unittest
import time
import numpy as np
from tinygrad import Tensor, Device, GlobalCounters, TinyJit
from tinygrad.helpers import getenv, colored
from tinygrad.nn.state import get_state_dict

# Llama 1B architecture (llama3.2:1b)
LLAMA_1B_CONFIG = {
    "dim": 2048,
    "n_layers": 22,
    "n_heads": 32,
    "n_kv_heads": 4,
    "norm_eps": 1e-05,
    "vocab_size": 32000,
    "hidden_dim": 5632,
}

BATCH_SIZE = 1
SEQ_LEN = 64
NUM_WARMUP = 15  # Increased to ensure JIT is fully compiled
NUM_TOKENS = 10

class TestLlama1BSpeed(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    """Initialize models once for all tests"""
    # Import torch and set single-threaded
    import torch
    torch.set_num_threads(1)
    cls.torch = torch

    # Create tinygrad model
    from extra.models.llama import Transformer
    print(f"Creating tinygrad Llama 1B model (dim={LLAMA_1B_CONFIG['dim']}, layers={LLAMA_1B_CONFIG['n_layers']})...")
    cls.tinygrad_model = Transformer(**LLAMA_1B_CONFIG, max_context=512)

    # Initialize with random weights (seed for reproducibility) - explicit float32
    np.random.seed(1337)
    from tinygrad.dtype import dtypes
    for v in get_state_dict(cls.tinygrad_model).values():
      # Force float32 to match PyTorch default
      target_dtype = dtypes.float32 if v.dtype in [dtypes.float16, dtypes.bfloat16] else v.dtype
      v.assign(Tensor.randn(*v.shape, dtype=target_dtype) * 0.02)

    # Create PyTorch model
    try:
      from transformers import LlamaForCausalLM, LlamaConfig
      print("Creating PyTorch Llama 1B model...")
      config = LlamaConfig(
          hidden_size=LLAMA_1B_CONFIG["dim"],
          num_hidden_layers=LLAMA_1B_CONFIG["n_layers"],
          num_attention_heads=LLAMA_1B_CONFIG["n_heads"],
          num_key_value_heads=LLAMA_1B_CONFIG["n_kv_heads"],
          intermediate_size=LLAMA_1B_CONFIG["hidden_dim"],
          vocab_size=LLAMA_1B_CONFIG["vocab_size"],
          rms_norm_eps=LLAMA_1B_CONFIG["norm_eps"],
          torch_dtype=cls.torch.float32,  # Explicit float32
      )
      cls.torch_model = LlamaForCausalLM(config)
      cls.torch_model.eval()
      cls.has_torch = True
    except ImportError:
      print("transformers not available, skipping PyTorch comparison")
      cls.has_torch = False

  def _benchmark_tinygrad(self):
    """Benchmark tinygrad inference speed"""
    print("\nTinyGrad Benchmark:")
    print(f"  Device: {Device.DEFAULT}")
    print(f"  Backend: CPU_LLVM={getenv('CPU_LLVM', 0)}, LLVM={getenv('LLVM', 0)}, CLANG={getenv('CLANG', 0)}")

    # Check dtype of model weights
    from tinygrad.nn.state import get_state_dict
    sample_weight = next(iter(get_state_dict(self.tinygrad_model).values()))
    print(f"  Model dtype: {sample_weight.dtype}")

    # JIT compile forward pass
    @TinyJit
    def forward(tokens, start_pos):
      return self.tinygrad_model(tokens, start_pos)

    tokens = Tensor([[1] * SEQ_LEN])

    # Warmup
    print(f"  Warming up ({NUM_WARMUP} tokens)...")
    for i in range(NUM_WARMUP):
      forward(tokens, i).realize()

    # Benchmark
    print(f"  Generating {NUM_TOKENS} tokens...")
    times = []
    GlobalCounters.reset()

    for i in range(NUM_TOKENS):
      Device[Device.DEFAULT].synchronize()
      st = time.perf_counter()
      forward(tokens, i).realize()
      Device[Device.DEFAULT].synchronize()
      et = (time.perf_counter() - st) * 1000
      times.append(et)

    avg_time = np.mean(times)
    min_time = np.min(times)

    print(f"  Average: {avg_time:.2f}ms/token ({1000/avg_time:.1f} tok/s)")
    print(f"  Min: {min_time:.2f}ms")
    print(f"  Total ops: {GlobalCounters.global_ops / 1e9:.2f}B")
    print(f"  Total mem: {GlobalCounters.global_mem / 1e9:.2f}GB")

    return avg_time, min_time

  def _benchmark_torch(self):
    """Benchmark PyTorch inference speed"""
    if not self.has_torch:
      return None, None

    print("\nPyTorch Benchmark:")
    print("  Device: CPU (single threaded)")

    tokens = self.torch.tensor([[1] * SEQ_LEN], dtype=self.torch.long)

    # Warmup
    print(f"  Warming up ({NUM_WARMUP} tokens)...")
    with self.torch.no_grad():
      for i in range(NUM_WARMUP):
        self.torch_model(tokens)

    # Benchmark
    print(f"  Generating {NUM_TOKENS} tokens...")
    times = []

    with self.torch.no_grad():
      for i in range(NUM_TOKENS):
        st = time.perf_counter()
        self.torch_model(tokens)
        et = (time.perf_counter() - st) * 1000
        times.append(et)

    avg_time = np.mean(times)
    min_time = np.min(times)

    print(f"  Average: {avg_time:.2f}ms/token ({1000/avg_time:.1f} tok/s)")
    print(f"  Min: {min_time:.2f}ms")

    return avg_time, min_time

  def test_llama1b_faster_than_torch(self):
    """Test that tinygrad is faster than PyTorch for Llama 1B inference"""
    print("\n" + "="*70)
    print("Llama 1B Speed Test: TinyGrad vs PyTorch on CPU")
    print("="*70)

    # Benchmark tinygrad
    tg_avg, tg_min = self._benchmark_tinygrad()

    # Benchmark PyTorch
    if not self.has_torch:
      self.skipTest("PyTorch transformers not available")
    pt_avg, pt_min = self._benchmark_torch()

    # Compare
    speedup = pt_avg / tg_avg
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"TinyGrad: {tg_avg:.2f}ms/token ({1000/tg_avg:.1f} tok/s)")
    print(f"PyTorch:  {pt_avg:.2f}ms/token ({1000/pt_avg:.1f} tok/s)")
    print(f"Speedup:  {speedup:.2f}x")

    if speedup > 1.0:
      print(f"\n{colored('✓ PASS', 'green')}: TinyGrad is {speedup:.2f}x faster than PyTorch!")
    else:
      print(f"\n{colored('✗ FAIL', 'red')}: TinyGrad is {1/speedup:.2f}x slower than PyTorch")

    print("="*70 + "\n")

    # Assert tinygrad is faster
    self.assertGreater(speedup, 1.0, f"TinyGrad must be faster than PyTorch (got {speedup:.2f}x)")

if __name__ == "__main__":
  unittest.main()
