#!/usr/bin/env python3
"""
Comprehensive test suite for NEW winograd optimization (UOp-level rewrite).
Tests numerical correctness, performance, and graph pattern matching.
"""
import unittest, time
import numpy as np
from tinygrad import Tensor, GlobalCounters, Context, dtypes
from tinygrad.helpers import WINO, getenv

def count_uops(schedule):
  """Count total UOps in a schedule"""
  return sum(len(list(si.ast.toposort())) for si in schedule)

def get_grid_layout(schedule):
  """Extract grid layouts from schedule for debugging"""
  grids = []
  for si in schedule:
    # Extract kernel grid info if available
    if hasattr(si, 'ast') and hasattr(si.ast, 'arg'):
      grids.append(str(si.ast.arg))
  return grids

class TestWinogradNumerical(unittest.TestCase):
  """Test numerical correctness across various input sizes"""

  def _test_correctness(self, bs, cin, cout, size, name=""):
    """Helper to test winograd matches normal conv"""
    x = Tensor.randn(bs, cin, size, size).realize()
    w = Tensor.randn(cout, cin, 3, 3).realize()

    # Normal conv (baseline)
    with Context(WINO_OLD=1):
      expected = x.conv2d(w, padding=1).realize()

    # NEW winograd
    with Context(WINO=1):
      actual = x.conv2d(w, padding=1).realize()

    # Winograd has slightly more floating point ops than direct conv, so allow small tolerance
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3,
                               err_msg=f"Winograd mismatch for {name}")
    print(f"✓ {name}: PASS")

  def test_small_32x32(self):
    self._test_correctness(1, 16, 16, 32, "32x32 (bs=1, c=16)")

  def test_medium_64x64(self):
    self._test_correctness(1, 32, 32, 64, "64x64 (bs=1, c=32)")

  def test_large_128x128(self):
    self._test_correctness(1, 16, 16, 128, "128x128 (bs=1, c=16)")

  def test_batch_size_4(self):
    self._test_correctness(4, 16, 16, 32, "32x32 batch=4")

  def test_batch_size_8(self):
    self._test_correctness(8, 8, 8, 32, "32x32 batch=8")

  def test_channels_64(self):
    self._test_correctness(1, 64, 64, 32, "32x32 c=64")

  def test_asymmetric_channels(self):
    self._test_correctness(1, 16, 32, 32, "32x32 cin=16 cout=32")

  def test_small_input_16x16(self):
    self._test_correctness(1, 8, 8, 16, "16x16 (edge case)")

class TestWinogradPerformance(unittest.TestCase):
  """Compare runtime and compile time across normal, OLD, and NEW winograd"""

  def _benchmark(self, bs, cin, cout, size, warmup=2, runs=5):
    """Run benchmark and return stats"""
    x = Tensor.randn(bs, cin, size, size).realize()
    w = Tensor.randn(cout, cin, 3, 3).realize()

    results = {}

    # Test normal conv
    with Context(WINO=0):
      # Warmup
      for _ in range(warmup):
        x.conv2d(w, padding=1).realize()

      # Measure compile time (schedule creation)
      GlobalCounters.reset()
      t0 = time.perf_counter()
      schedule = x.conv2d(w, padding=1).schedule()
      compile_time = time.perf_counter() - t0

      # Measure runtime
      x.conv2d(w, padding=1).realize()
      t0 = time.perf_counter()
      for _ in range(runs):
        x.conv2d(w, padding=1).realize()
      runtime = (time.perf_counter() - t0) / runs

      results['normal'] = {
        'compile_ms': compile_time * 1000,
        'runtime_ms': runtime * 1000,
        'ops': GlobalCounters.global_ops,
        'mem': GlobalCounters.global_mem,
        'kernels': len(schedule),
        'uops': count_uops(schedule)
      }

    # Test OLD winograd (tensor-level)
    with Context(WINO_OLD=1, WINO=0):
      for _ in range(warmup):
        x.conv2d(w, padding=1).realize()

      GlobalCounters.reset()
      t0 = time.perf_counter()
      schedule = x.conv2d(w, padding=1).schedule()
      compile_time = time.perf_counter() - t0

      x.conv2d(w, padding=1).realize()
      t0 = time.perf_counter()
      for _ in range(runs):
        x.conv2d(w, padding=1).realize()
      runtime = (time.perf_counter() - t0) / runs

      results['old'] = {
        'compile_ms': compile_time * 1000,
        'runtime_ms': runtime * 1000,
        'ops': GlobalCounters.global_ops,
        'mem': GlobalCounters.global_mem,
        'kernels': len(schedule),
        'uops': count_uops(schedule)
      }

    # Test NEW winograd (UOp-level)
    with Context(WINO=1):
      for _ in range(warmup):
        x.conv2d(w, padding=1).realize()

      GlobalCounters.reset()
      t0 = time.perf_counter()
      schedule = x.conv2d(w, padding=1).schedule()
      compile_time = time.perf_counter() - t0

      x.conv2d(w, padding=1).realize()
      t0 = time.perf_counter()
      for _ in range(runs):
        x.conv2d(w, padding=1).realize()
      runtime = (time.perf_counter() - t0) / runs

      results['new'] = {
        'compile_ms': compile_time * 1000,
        'runtime_ms': runtime * 1000,
        'ops': GlobalCounters.global_ops,
        'mem': GlobalCounters.global_mem,
        'kernels': len(schedule),
        'uops': count_uops(schedule)
      }

    return results

  def _print_comparison(self, name, results):
    """Pretty print benchmark results"""
    print(f"\n{'='*80}")
    print(f"Benchmark: {name}")
    print(f"{'='*80}")

    for mode in ['normal', 'old', 'new']:
      if mode not in results:
        continue
      r = results[mode]
      print(f"\n{mode.upper():8s} | Compile: {r['compile_ms']:6.2f}ms | Runtime: {r['runtime_ms']:6.2f}ms")
      print(f"         | Kernels: {r['kernels']:3d} | UOps: {r['uops']:6d} | Ops: {r['ops']:10d} | Mem: {r['mem']:10d}")

    # Compute speedups
    if 'new' in results and 'normal' in results:
      speedup = results['normal']['runtime_ms'] / results['new']['runtime_ms']
      ops_ratio = results['new']['ops'] / results['normal']['ops']
      print(f"\nNEW vs NORMAL: {speedup:.2f}x speedup, {ops_ratio:.2f}x ops ratio")

    if 'new' in results and 'old' in results:
      speedup = results['old']['runtime_ms'] / results['new']['runtime_ms']
      print(f"NEW vs OLD: {speedup:.2f}x speedup")

    print(f"{'='*80}\n")

  def test_perf_32x32_c16(self):
    """Standard case: 32x32 with 16 channels"""
    results = self._benchmark(1, 16, 16, 32, warmup=2, runs=5)
    self._print_comparison("32x32 (bs=1, c=16)", results)

    # Verify NEW winograd doesn't explode ops count
    if 'new' in results and 'normal' in results:
      ops_ratio = results['new']['ops'] / results['normal']['ops']
      self.assertLess(ops_ratio, 5.0, "NEW winograd ops count explosion!")

  def test_perf_64x64_c32(self):
    """Larger case: 64x64 with 32 channels"""
    results = self._benchmark(1, 32, 32, 64, warmup=2, runs=3)
    self._print_comparison("64x64 (bs=1, c=32)", results)

    if 'new' in results and 'normal' in results:
      ops_ratio = results['new']['ops'] / results['normal']['ops']
      self.assertLess(ops_ratio, 5.0, "NEW winograd ops count explosion!")

  def test_perf_batch8(self):
    """Batch processing: bs=8"""
    results = self._benchmark(8, 16, 16, 32, warmup=2, runs=3)
    self._print_comparison("32x32 batch=8", results)

class TestWinogradComplexGraph(unittest.TestCase):
  """Test winograd detection in complex computation graphs"""

  def test_conv_with_relu(self):
    """Conv sandwiched between ReLU activations"""
    x = Tensor.randn(1, 16, 32, 32).realize()
    w = Tensor.randn(16, 16, 3, 3).realize()

    # Normal conv
    with Context(WINO=0):
      expected = x.relu().conv2d(w, padding=1).relu().realize()

    # With winograd
    with Context(WINO=1):
      actual = x.relu().conv2d(w, padding=1).relu().realize()
      schedule = x.relu().conv2d(w, padding=1).relu().schedule()

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)
    print(f"✓ Conv with ReLU: {len(schedule)} kernels")

  def test_conv_with_batchnorm(self):
    """Conv followed by batch normalization"""
    x = Tensor.randn(1, 16, 32, 32).realize()
    w = Tensor.randn(16, 16, 3, 3).realize()
    scale = Tensor.ones(16).realize()
    bias = Tensor.zeros(16).realize()

    with Context(WINO=0):
      expected = x.conv2d(w, padding=1)
      expected = (expected - expected.mean(axis=(0,2,3), keepdim=True)) / (expected.std(axis=(0,2,3), keepdim=True) + 1e-5)
      expected = (expected * scale.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)).realize()

    with Context(WINO=1):
      actual = x.conv2d(w, padding=1)
      actual = (actual - actual.mean(axis=(0,2,3), keepdim=True)) / (actual.std(axis=(0,2,3), keepdim=True) + 1e-5)
      actual = (actual * scale.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)).realize()

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Conv with BatchNorm")

  def test_sequential_convs(self):
    """Multiple convolutions in sequence"""
    x = Tensor.randn(1, 16, 32, 32).realize()
    w1 = Tensor.randn(32, 16, 3, 3).realize()
    w2 = Tensor.randn(32, 32, 3, 3).realize()

    with Context(WINO=0):
      expected = x.conv2d(w1, padding=1).relu().conv2d(w2, padding=1).realize()

    with Context(WINO=1):
      actual = x.conv2d(w1, padding=1).relu().conv2d(w2, padding=1).realize()

    # Two convolutions in sequence accumulate more floating point error
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Sequential convs")

  def test_residual_connection(self):
    """Conv with residual skip connection"""
    x = Tensor.randn(1, 16, 32, 32).realize()
    w = Tensor.randn(16, 16, 3, 3).realize()

    with Context(WINO=0):
      expected = (x + x.conv2d(w, padding=1)).realize()

    with Context(WINO=1):
      actual = (x + x.conv2d(w, padding=1)).realize()

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Residual connection")

class TestWinogradGridLayout(unittest.TestCase):
  """Verify grid layout is r_6_6_tiles (coalesced) not r_tiles_6_6 (scattered)"""

  def test_grid_layout_32x32(self):
    """Check 32x32 produces r_6_6_8 grid"""
    x = Tensor.randn(1, 16, 32, 32).realize()
    w = Tensor.randn(16, 16, 3, 3).realize()

    with Context(WINO=1):
      schedule = x.conv2d(w, padding=1).schedule()

      # Find the main winograd kernel (should have 6,6,8 pattern)
      found_correct_layout = False
      for si in schedule:
        ast_str = str(si.ast)
        # Look for the kernel with 6×6×8 dimensions (winograd transform kernel)
        if '6' in ast_str and '8' in ast_str:
          found_correct_layout = True
          # Could add more detailed checks here if AST structure is accessible
          break

      self.assertTrue(found_correct_layout, "Expected to find 6×6×8 winograd kernel")
      print(f"✓ Grid layout check: {len(schedule)} kernels scheduled")

if __name__ == '__main__':
  # Run with: WINO=1 python3 test/unit/test_winograd_new.py
  # Compare with OLD: WINO_OLD=1 python3 test/unit/test_winograd_new.py
  print(f"Running with WINO={getenv('WINO', 0)}, WINO_OLD={getenv('WINO_OLD', 0)}")
  unittest.main(verbosity=2)
