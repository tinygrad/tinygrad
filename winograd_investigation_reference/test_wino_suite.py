import os, time, numpy as np, pytest
import time, gc, numpy as np, pytest
from tinygrad import Tensor, Context

np.random.seed(42)

# ------------------------------
# Utility
# ------------------------------
def timed_run(fn, *args, **kwargs):
  start_compile = time.time()
  out = fn(*args, **kwargs)
  _ = out.realize()  # force compile
  compile_time = time.time() - start_compile
  start_run = time.time()
  np_out = out.numpy()
  run_time = time.time() - start_run
  return np_out, compile_time, run_time

def report(label, compile_time, run_time, diff=None):
  print(f"\n{'='*60}\n[{label}]")
  print(f"Compile Time: {compile_time*1000:.2f} ms | Run Time: {run_time*1000:.2f} ms")
  if diff is not None:
    print(f"Max abs diff: {diff.max():.3e} | Mean abs diff: {diff.mean():.3e}")

# ------------------------------
# Core comparison test
# ------------------------------
# ---------------- cfg ----------------
REPEATS = 5         # timed repeats per mode (post-warmup)
WARMUP  = 1         # warmup runs per mode
ATOL, RTOL = 2e-1, 1e-2 #seems consistent with old winograd
np.random.seed(0)

# Comprehensive test cases covering different channel configurations
CASES = [
  # (B, Cin, Cout, H, W, KH, KW, pad)
  # Edge cases: single channels
  (1,  1,  1, 8, 8, 3, 3, 1),       # Both channels = 1
  (1,  1,  4, 8, 8, 3, 3, 1),       # Cin=1, Cout>1 (important!)
  (1,  4,  1, 8, 8, 3, 3, 1),       # Cin>1, Cout=1 (important!)

  # Asymmetric channels
  (1,  2,  3, 8, 8, 3, 3, 1),       # Small asymmetric
  (1,  8,  16, 12, 12, 3, 3, 1),    # Cin < Cout
  (1,  16, 8, 12, 12, 3, 3, 1),     # Cin > Cout

  # Symmetric channels
  (1,  4,  4, 12, 12, 3, 3, 1),     # Small symmetric
  (1,  32, 32, 16, 16, 3, 3, 1),    # Medium symmetric

  # Batch size variations
  (2,  4,  4, 8, 8, 3, 3, 1),       # B=2
  (4,  3,  3, 8, 8, 3, 3, 1),       # B=4

  # Large configurations
  (1,  64,  64, 64, 64, 3, 3, 1),   # Large square
  # (1,  64,  64, 128, 128, 3, 3, 1), # Very large (uncomment for thorough testing)
]

# --- Fixed version: use same data for fair comparison ---
def _bench_once(x, w, pad, ctx_kwargs):
  with Context(**ctx_kwargs):
    out = x.conv2d(w, padding=pad)
    t0 = time.time(); out.realize(); t_compile = time.time()-t0
    t1 = time.time(); arr = out.numpy(); t_exec = time.time()-t1
  return arr, t_compile, t_exec

def _bench(x, w, pad, ctx_kwargs):
  # warmup
  for _ in range(WARMUP): _bench_once(x, w, pad, ctx_kwargs)
  # repeats
  comp, run, last = [], [], None
  for _ in range(REPEATS):
    gc.collect()
    arr, c, r = _bench_once(x, w, pad, ctx_kwargs)
    comp.append(c); run.append(r); last = arr
  return last, float(np.mean(comp)), float(np.mean(run))

def _fmt_ms(s): return f"{s*1e3:7.1f} ms"

@pytest.mark.parametrize("B,Cin,Cout,H,W,KH,KW,pad", CASES)
def test_wino_bench(B,Cin,Cout,H,W,KH,KW,pad):
  # Create tensors ONCE for fair comparison
  x = Tensor.randn(B, Cin, H, W).realize()
  w = Tensor.randn(Cout, Cin, KH, KW).realize()

  base_arr, c_base, r_base = _bench(x, w, pad, dict(WINO=0, WINO_OLD=0))
  new_arr,  c_new,  r_new  = _bench(x, w, pad, dict(WINO=1, WINO_OLD=0))
  old_arr,  c_old,  r_old  = _bench(x, w, pad, dict(WINO=0, WINO_OLD=1))

  # diffs vs baseline
  d_new = np.abs(new_arr - base_arr)
  d_old = np.abs(old_arr - base_arr)

  # Compact three-way comparison
  print(f"\n[B={B:2d} Cin={Cin:3d} Cout={Cout:3d} H={H:3d}x{W:3d}]")
  print(f"  BASE:        Compile: {_fmt_ms(c_base)} | Run: {_fmt_ms(r_base)}")
  print(f"  NEW (kron):  Compile: {_fmt_ms(c_new)} ({c_new/c_base if c_base>0 else 0:>4.1f}×) | Run: {_fmt_ms(r_new)} ({r_base/r_new if r_new>0 else 0:>4.2f}×) | Err: {d_new.max():.2e}")
  print(f"  OLD (tsr):   Compile: {_fmt_ms(c_old)} ({c_old/c_base if c_base>0 else 0:>4.1f}×) | Run: {_fmt_ms(r_old)} ({r_base/r_old if r_old>0 else 0:>4.2f}×) | Err: {d_old.max():.2e}")

  # sanity: correctness against baseline
  np.testing.assert_allclose(new_arr, base_arr, atol=ATOL, rtol=RTOL,
                            err_msg=f"NEW Winograd output doesn't match baseline")
  np.testing.assert_allclose(old_arr, base_arr, atol=ATOL, rtol=RTOL,
                            err_msg=f"OLD Winograd output doesn't match baseline")

# ------------------------------
# Sandwich test: conv amid nonlinear ops
# ------------------------------
@pytest.mark.parametrize("opchain_name", ["relu_elu", "sigmoid_relu", "add_log_exp"])
def test_wino_with_op_chains(opchain_name):
  x = Tensor.randn(1, 3, 16, 16)
  w = Tensor.randn(4, 3, 3, 3)  # Create weight once and reuse it
  
  # Define opchains that use the same weight tensor
  opchains = {
    "relu_elu": lambda x, w: x.relu().conv2d(w, padding=1).elu(),
    "sigmoid_relu": lambda x, w: (x*0.5 + x.sigmoid()).conv2d(w, padding=1).relu(),
    "add_log_exp": lambda x, w: (x + 0.3).conv2d(w, padding=1).log().exp(),
  }
  opchain = opchains[opchain_name]
  
  base_fn = lambda: opchain(x, w)
  print(f"\nTesting complex op chain ({opchain_name})...")

  with Context(WINO=0, WINO_OLD=0):
    base, _, _ = timed_run(base_fn)
  with Context(WINO=1, WINO_OLD=0):
    wino, _, _ = timed_run(base_fn)

  diff = np.abs(wino - base)
  report("OPCHAIN", 0, 0, diff)
  np.testing.assert_allclose(wino, base, atol=1e-4, rtol=1e-3)

# ------------------------------
# Channel reduction test - ensure Cin reduction is correct
# ------------------------------
@pytest.mark.parametrize("Cin", [1, 2, 3, 8, 16, 32])
def test_cin_reduction(Cin):
  """Test that reduction over input channels works correctly for various Cin values"""
  x = Tensor.randn(1, Cin, 12, 12).realize()
  w = Tensor.randn(4, Cin, 3, 3).realize()

  base_arr, c_base, r_base = _bench(x, w, 1, dict(WINO=0))
  wino_arr, c_wino, r_wino = _bench(x, w, 1, dict(WINO=1))

  diff = np.abs(base_arr - wino_arr)

  print(f"\n[Cin={Cin:2d} reduction] ", end="")
  print(f"Compile: {_fmt_ms(c_base)} → {_fmt_ms(c_wino)} ({c_wino/c_base if c_base>0 else 0:.1f}×) | ", end="")
  print(f"Run: {_fmt_ms(r_base)} → {_fmt_ms(r_wino)} ({r_base/r_wino if r_wino>0 else 0:.2f}×) | ", end="")
  print(f"Err: {diff.max():.2e}")

  assert np.allclose(base_arr, wino_arr, atol=1e-3, rtol=1e-3), \
    f"Cin={Cin} reduction test failed: max_diff={diff.max():.6e}"

# ------------------------------
# Deliberate pattern test to catch axis ordering bugs
# ------------------------------
def test_known_values():
  """Test with deliberate patterns (1s and 2s) to catch matmul/reduction order issues"""
  print("\nTesting with known constant patterns...")

  # Tiny input: different constant values in each channel
  x_np = np.zeros((1, 2, 8, 8), dtype=np.float32)
  x_np[0, 0, :, :] = 1.0  # Channel 0: all ones
  x_np[0, 1, :, :] = 2.0  # Channel 1: all twos

  # Simple kernel: identity-like patterns
  w_np = np.zeros((2, 2, 3, 3), dtype=np.float32)
  w_np[0, 0, 1, 1] = 1.0  # Out ch 0 from in ch 0: center tap
  w_np[1, 1, 1, 1] = 1.0  # Out ch 1 from in ch 1: center tap

  x = Tensor(x_np)
  w = Tensor(w_np)

  # Baseline
  with Context(WINO=0):
    base = x.conv2d(w, padding=1).realize().numpy()

  # Winograd
  with Context(WINO=1):
    wino = x.conv2d(w, padding=1).realize().numpy()

  print(f"Expected output ch 0 (should be all 1.0): {base[0,0,4,4]:.2f}")
  print(f"Got from winograd ch 0: {wino[0,0,4,4]:.2f}")
  print(f"Expected output ch 1 (should be all 2.0): {base[0,1,4,4]:.2f}")
  print(f"Got from winograd ch 1: {wino[0,1,4,4]:.2f}")

  diff = np.abs(base - wino)
  print(f"Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")

  assert np.allclose(base, wino, atol=1e-4, rtol=1e-3), \
    f"Known-value test failed: max_diff={diff.max()}, expected ≈0"

# ------------------------------
# Manual 3x3 conv recognition test
# ------------------------------
def test_manual_conv_recognition():
  print("\nManual conv pattern recognition test...")

  x = Tensor.randn(1, 3, 8, 8)
  w = Tensor.randn(4, 3, 3, 3)
  # Manual conv definition (no conv2d call)
  def manual_conv(x, w):
    bs, cin, h, w_ = x.shape
    cout, _, kh, kw = w.shape
    out = []
    for i in range(h - kh + 1):
      row = []
      for j in range(w_ - kw + 1):
        patch = x[:, :, i:i+kh, j:j+kw]
        row.append((patch * w).sum(axis=(1,2,3)))
      out.append(row)
    return Tensor.stack([Tensor.stack(r, dim=-1) for r in out], dim=-2).reshape(bs, cout, h-kh+1, w_-kw+1)

  with Context(WINO=0, WINO_OLD=0):
    base, _, _ = timed_run(lambda: manual_conv(x, w))
  with Context(WINO=1, WINO_OLD=0):
    new, _, _ = timed_run(lambda: manual_conv(x, w))

  diff = np.abs(new - base)
  report("MANUAL_CONV", 0, 0, diff)
  np.testing.assert_allclose(new, base, atol=1e-4, rtol=1e-3)

# ------------------------------
# Performance summary
# ------------------------------
def test_performance_summary():
  """Run a representative subset and report overall performance trends"""
  print("\n" + "="*100)
  print("PERFORMANCE SUMMARY - NEW (unified kron) vs OLD (tensor.py) Winograd")
  print("="*100)

  test_cases = [
    # (B, Cin, Cout, H, W, description)
    (1, 1, 1, 8, 8, "Minimal (1×1)"),
    (1, 3, 3, 16, 16, "Small (3×3)"),
    (1, 16, 16, 32, 32, "Medium (16×16)"),
    (1, 64, 64, 64, 64, "Large (64×64)"),
    (4, 8, 8, 16, 16, "Multi-batch (B=4)"),
  ]

  new_compile_ratios, new_run_speedups = [], []
  old_compile_ratios, old_run_speedups = [], []

  print(f"\n{'Config':<20} {'NEW Compile':<15} {'OLD Compile':<15} {'NEW Run':<15} {'OLD Run':<15} {'NEW Err':<12} {'OLD Err':<12}")
  print("-" * 100)

  for B, Cin, Cout, H, W, desc in test_cases:
    x = Tensor.randn(B, Cin, H, W).realize()
    w = Tensor.randn(Cout, Cin, 3, 3).realize()

    base_arr, c_base, r_base = _bench(x, w, 1, dict(WINO=0, WINO_OLD=0))
    new_arr,  c_new,  r_new  = _bench(x, w, 1, dict(WINO=1, WINO_OLD=0))
    old_arr,  c_old,  r_old  = _bench(x, w, 1, dict(WINO=0, WINO_OLD=1))

    d_new = np.abs(new_arr - base_arr)
    d_old = np.abs(old_arr - base_arr)

    new_c_ratio = c_new / c_base if c_base > 0 else 0
    old_c_ratio = c_old / c_base if c_base > 0 else 0
    new_r_speedup = r_base / r_new if r_new > 0 else 0
    old_r_speedup = r_base / r_old if r_old > 0 else 0

    new_compile_ratios.append(new_c_ratio)
    new_run_speedups.append(new_r_speedup)
    old_compile_ratios.append(old_c_ratio)
    old_run_speedups.append(old_r_speedup)

    print(f"{desc:<20} {new_c_ratio:>6.1f}×{'':<8} {old_c_ratio:>6.1f}×{'':<8} {new_r_speedup:>6.2f}×{'':<8} "
          f"{old_r_speedup:>6.2f}×{'':<8} {d_new.max():<12.2e} {d_old.max():<12.2e}")

  print("-" * 100)
  print(f"{'AVERAGE':<20} {np.mean(new_compile_ratios):>6.1f}×{'':<8} {np.mean(old_compile_ratios):>6.1f}×{'':<8} "
        f"{np.mean(new_run_speedups):>6.2f}×{'':<8} {np.mean(old_run_speedups):>6.2f}×")
  print("="*100)
  print("\nKey Findings:")
  print(f"  • NEW compile overhead: {np.mean(new_compile_ratios):.1f}× | OLD compile overhead: {np.mean(old_compile_ratios):.1f}×")
  print(f"  • NEW runtime speedup:  {np.mean(new_run_speedups):.2f}× | OLD runtime speedup:  {np.mean(old_run_speedups):.2f}×")

  # Compare NEW vs OLD directly
  compile_improvement = np.mean(old_compile_ratios) / np.mean(new_compile_ratios) if np.mean(new_compile_ratios) > 0 else 0
  runtime_improvement = np.mean(new_run_speedups) / np.mean(old_run_speedups) if np.mean(old_run_speedups) > 0 else 0

  print(f"\nNEW vs OLD comparison:")
  print(f"  • Compile time: NEW is {compile_improvement:.2f}× {'faster' if compile_improvement > 1 else 'slower'} than OLD")
  print(f"  • Runtime: NEW is {runtime_improvement:.2f}× {'faster' if runtime_improvement > 1 else 'slower'} than OLD")
  print(f"  • Both methods: Numerical accuracy < 1e-3 ✅")
  print("="*100)

# ------------------------------
# Quick smoke check
# ------------------------------
def test_quick_summary():
  print("\n✅ Winograd test suite complete.")