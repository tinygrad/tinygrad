#!/usr/bin/env python3
"""SQTT InstOp discovery from tinygrad-generated kernels.

Runs various tinygrad operations and captures SQTT traces to find new InstOp values.

Requires profiling enabled:
  echo 'profile_standard' | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level

Run with: DEBUG=1 python extra/assembly/amd/test/discover_instops_tensor.py
For full traces: DEBUG=2 python extra/assembly/amd/test/discover_instops_tensor.py
"""
import os
os.environ["SQTT"] = "1"
os.environ["PROFILE"] = "1"
os.environ["SQTT_LIMIT_SE"] = "2"  # Force work to traced SE only
os.environ["SQTT_TOKEN_EXCLUDE"] = "3784"  # Exclude noisy packet types

from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import DEBUG, colored
from tinygrad.runtime.ops_amd import ProfileSQTTEvent, SQTT_SIMD_SEL

from extra.assembly.amd.sqtt import InstOp, decode, INST, WAVESTART, WAVEEND

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_inst_ops_from_blobs(blobs: list[bytes]) -> set[int]:
  """Extract all InstOp values from SQTT blobs."""
  ops = set()
  for blob in blobs:
    packets = decode(blob)
    in_wave = False
    for p in packets:
      if isinstance(p, WAVESTART):
        in_wave = True
      if in_wave and isinstance(p, INST):
        ops.add(p.op if isinstance(p.op, int) else p.op.value)
      if isinstance(p, WAVEEND):
        in_wave = False
  return ops

def run_and_capture(fn, attempts: int = 5) -> tuple[set[int], list[bytes]]:
  """Run a function multiple times and collect SQTT traces."""
  dev = Device["AMD"]
  all_ops = set()
  all_blobs = []
  SQTT_SIMD_SEL.value = 0

  for _ in range(attempts):
    dev.profile_events.clear()
    fn()
    blobs = [ev.blob for ev in dev.profile_events if isinstance(ev, ProfileSQTTEvent)]
    ops = get_inst_ops_from_blobs(blobs)
    all_ops.update(ops)
    all_blobs.extend(blobs)

  return all_ops, all_blobs

# ═══════════════════════════════════════════════════════════════════════════════
# TENSOR OPERATIONS TO TEST
# ═══════════════════════════════════════════════════════════════════════════════

TENSOR_TESTS: dict[str, tuple[str, callable]] = {
  # Basic arithmetic
  "add_f32": ("tensor add float32", lambda: (Tensor.rand(1024) + Tensor.rand(1024)).realize()),
  "mul_f32": ("tensor mul float32", lambda: (Tensor.rand(1024) * Tensor.rand(1024)).realize()),
  "sub_f32": ("tensor sub float32", lambda: (Tensor.rand(1024) - Tensor.rand(1024)).realize()),
  "div_f32": ("tensor div float32", lambda: (Tensor.rand(1024) / (Tensor.rand(1024) + 0.1)).realize()),

  # Transcendental
  "exp_f32": ("tensor exp float32", lambda: Tensor.rand(1024).exp().realize()),
  "log_f32": ("tensor log float32", lambda: (Tensor.rand(1024) + 0.1).log().realize()),
  "sqrt_f32": ("tensor sqrt float32", lambda: Tensor.rand(1024).sqrt().realize()),
  "sin_f32": ("tensor sin float32", lambda: Tensor.rand(1024).sin().realize()),
  "cos_f32": ("tensor cos float32", lambda: Tensor.rand(1024).cos().realize()),
  "tanh_f32": ("tensor tanh float32", lambda: Tensor.rand(1024).tanh().realize()),
  "sigmoid_f32": ("tensor sigmoid float32", lambda: Tensor.rand(1024).sigmoid().realize()),

  # Reductions
  "sum_f32": ("tensor sum float32", lambda: Tensor.rand(1024).sum().realize()),
  "max_f32": ("tensor max float32", lambda: Tensor.rand(1024).max().realize()),
  "mean_f32": ("tensor mean float32", lambda: Tensor.rand(1024).mean().realize()),

  # Matmul - small
  "matmul_small": ("matmul 32x32", lambda: (Tensor.rand(32, 32) @ Tensor.rand(32, 32)).realize()),

  # Matmul - medium (might use WMMA)
  "matmul_medium": ("matmul 128x128", lambda: (Tensor.rand(128, 128) @ Tensor.rand(128, 128)).realize()),

  # Matmul - larger (more likely to use WMMA)
  "matmul_large": ("matmul 256x256", lambda: (Tensor.rand(256, 256) @ Tensor.rand(256, 256)).realize()),

  # Different dtypes
  "add_f16": ("tensor add float16", lambda: (Tensor.rand(1024, dtype=dtypes.float16) + Tensor.rand(1024, dtype=dtypes.float16)).realize()),
  "mul_f16": ("tensor mul float16", lambda: (Tensor.rand(1024, dtype=dtypes.float16) * Tensor.rand(1024, dtype=dtypes.float16)).realize()),
  "matmul_f16": ("matmul float16 128x128", lambda: (Tensor.rand(128, 128, dtype=dtypes.float16) @ Tensor.rand(128, 128, dtype=dtypes.float16)).realize()),

  # Integer ops
  "add_i32": ("tensor add int32", lambda: (Tensor.randint(1024, high=1000) + Tensor.randint(1024, high=1000)).realize()),
  "mul_i32": ("tensor mul int32", lambda: (Tensor.randint(1024, high=100) * Tensor.randint(1024, high=100)).realize()),

  # Bitwise
  "and_i32": ("tensor bitwise and", lambda: (Tensor.randint(1024, high=1000) & Tensor.randint(1024, high=1000)).realize()),
  "or_i32": ("tensor bitwise or", lambda: (Tensor.randint(1024, high=1000) | Tensor.randint(1024, high=1000)).realize()),
  "xor_i32": ("tensor bitwise xor", lambda: (Tensor.randint(1024, high=1000) ^ Tensor.randint(1024, high=1000)).realize()),
  "lshift_i32": ("tensor left shift", lambda: (Tensor.randint(1024, high=1000) << 2).realize()),
  "rshift_i32": ("tensor right shift", lambda: (Tensor.randint(1024, high=1000) >> 2).realize()),

  # Comparisons
  "cmp_eq": ("tensor compare eq", lambda: (Tensor.rand(1024) == 0.5).realize()),
  "cmp_lt": ("tensor compare lt", lambda: (Tensor.rand(1024) < 0.5).realize()),
  "cmp_gt": ("tensor compare gt", lambda: (Tensor.rand(1024) > 0.5).realize()),

  # Where/select
  "where": ("tensor where", lambda: Tensor.rand(1024).where(Tensor.rand(1024), Tensor.rand(1024)).realize()),

  # Reshaping/movement (may not generate interesting ops but let's check)
  "reshape": ("tensor reshape", lambda: Tensor.rand(32, 32).reshape(16, 64).realize()),
  "permute": ("tensor permute", lambda: Tensor.rand(32, 32).permute(1, 0).contiguous().realize()),
  "expand": ("tensor expand", lambda: Tensor.rand(1, 32).expand(32, 32).contiguous().realize()),

  # Pad
  "pad": ("tensor pad", lambda: Tensor.rand(30, 30).pad(((1, 1), (1, 1))).realize()),

  # Conv2D - small
  "conv2d_small": ("conv2d 3x3", lambda: Tensor.rand(1, 3, 32, 32).conv2d(Tensor.rand(8, 3, 3, 3)).realize()),

  # Conv2D - larger
  "conv2d_medium": ("conv2d 3x3 64ch", lambda: Tensor.rand(1, 64, 32, 32).conv2d(Tensor.rand(64, 64, 3, 3)).realize()),

  # Pooling
  "maxpool": ("max pool 2x2", lambda: Tensor.rand(1, 3, 32, 32).max_pool2d((2, 2)).realize()),
  "avgpool": ("avg pool 2x2", lambda: Tensor.rand(1, 3, 32, 32).avg_pool2d((2, 2)).realize()),

  # Softmax
  "softmax": ("softmax", lambda: Tensor.rand(32, 128).softmax().realize()),

  # LayerNorm-like
  "layernorm": ("layer norm pattern", lambda: _layernorm(Tensor.rand(32, 128))),

  # BatchNorm-like
  "batchnorm": ("batch norm pattern", lambda: _batchnorm(Tensor.rand(1, 64, 32, 32))),

  # Dropout-like (during training)
  "dropout": ("dropout pattern", lambda: (Tensor.rand(1024) * (Tensor.rand(1024) > 0.5)).realize()),

  # Cast operations
  "cast_f32_to_f16": ("cast f32->f16", lambda: Tensor.rand(1024).cast(dtypes.float16).realize()),
  "cast_f16_to_f32": ("cast f16->f32", lambda: Tensor.rand(1024, dtype=dtypes.float16).cast(dtypes.float32).realize()),
  "cast_f32_to_i32": ("cast f32->i32", lambda: (Tensor.rand(1024) * 100).cast(dtypes.int32).realize()),
  "cast_i32_to_f32": ("cast i32->f32", lambda: Tensor.randint(1024, high=100).cast(dtypes.float32).realize()),

  # Clamp/clip
  "clamp": ("tensor clamp", lambda: Tensor.rand(1024).clamp(0.2, 0.8).realize()),

  # Abs/neg
  "abs": ("tensor abs", lambda: (Tensor.rand(1024) - 0.5).abs().realize()),
  "neg": ("tensor neg", lambda: (-Tensor.rand(1024)).realize()),

  # Reciprocal
  "recip": ("tensor reciprocal", lambda: (Tensor.rand(1024) + 0.1).reciprocal().realize()),

  # Power
  "pow2": ("tensor pow 2", lambda: (Tensor.rand(1024) ** 2).realize()),
  "pow3": ("tensor pow 3", lambda: (Tensor.rand(1024) ** 3).realize()),
}

def _layernorm(x: Tensor) -> Tensor:
  """Simple layer normalization pattern."""
  mean = x.mean(axis=-1, keepdim=True)
  var = ((x - mean) ** 2).mean(axis=-1, keepdim=True)
  return ((x - mean) / (var + 1e-5).sqrt()).realize()

def _batchnorm(x: Tensor) -> Tensor:
  """Simple batch normalization pattern."""
  mean = x.mean(axis=(0, 2, 3), keepdim=True)
  var = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdim=True)
  return ((x - mean) / (var + 1e-5).sqrt()).realize()

# ═══════════════════════════════════════════════════════════════════════════════
# DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def discover_all_instops() -> tuple[dict[int, set[str]], dict[str, Exception]]:
  """Run all tensor tests and collect InstOp values."""
  discovered: dict[int, set[str]] = {}
  failures: dict[str, Exception] = {}

  for test_name, (desc, fn) in TENSOR_TESTS.items():
    try:
      ops, blobs = run_and_capture(fn)

      for op in ops:
        if op not in discovered:
          discovered[op] = set()
        discovered[op].add(test_name)

      if DEBUG >= 1:
        status = colored("✓", "green") if ops else colored("∅", "yellow")
        ops_str = ", ".join(hex(op) for op in sorted(ops)) if ops else "none"
        print(f"  {status} {test_name:25s} [{desc:25s}] ops=[{ops_str}]")

      if DEBUG >= 2 and blobs:
        # Show first wave trace
        for blob in blobs[:1]:
          packets = decode(blob)
          print(f"    First blob: {len(blob)} bytes, {len(packets)} packets")

    except Exception as e:
      failures[test_name] = e
      if DEBUG >= 1:
        print(f"  {colored('✗', 'red')} {test_name:25s} FAILED: {e}")

  return discovered, failures


def print_summary(discovered: dict[int, set[str]], failures: dict[str, Exception]) -> None:
  """Print discovery summary."""
  known_ops = {e.value for e in InstOp}
  discovered_ops = set(discovered.keys())

  print("\n" + "=" * 70)
  print("DISCOVERED INSTOP VALUES FROM TINYGRAD KERNELS")
  print("=" * 70)

  for op in sorted(discovered_ops):
    try:
      name = InstOp(op).name
      status = colored("known", "green")
    except ValueError:
      name = "UNKNOWN"
      status = colored("NEW!", "yellow")

    sources = ", ".join(sorted(discovered[op]))
    # Truncate sources if too long
    if len(sources) > 60:
      sources = sources[:57] + "..."
    print(f"  0x{op:02x} {name:20s} ({status}) <- {sources}")

  # New values to add
  new_ops = discovered_ops - known_ops
  if new_ops:
    print("\n" + "=" * 70)
    print(colored("NEW INSTOP VALUES TO ADD TO ENUM", "yellow"))
    print("=" * 70)
    for op in sorted(new_ops):
      sources = ", ".join(sorted(discovered[op]))
      print(f"  0x{op:02x}: discovered from [{sources}]")

  # Missing from enum (not discovered)
  missing = known_ops - discovered_ops
  if missing:
    print("\n" + "=" * 70)
    print("ENUM VALUES NOT DISCOVERED (may need specific instruction patterns)")
    print("=" * 70)
    for op in sorted(missing):
      print(f"  0x{op:02x} {InstOp(op).name}")

  # Stats
  print("\n" + "=" * 70)
  print("STATISTICS")
  print("=" * 70)
  print(f"  Tests run:      {len(TENSOR_TESTS)}")
  print(f"  Tests passed:   {len(TENSOR_TESTS) - len(failures)}")
  print(f"  Tests failed:   {len(failures)}")
  print(f"  Known ops:      {len(known_ops)}")
  print(f"  Discovered:     {len(discovered_ops)}")
  if known_ops:
    coverage = len(discovered_ops & known_ops)
    print(f"  Coverage:       {coverage}/{len(known_ops)} ({100*coverage//len(known_ops)}%)")
  print(f"  New ops found:  {len(new_ops)}")

  if failures:
    print("\n" + "=" * 70)
    print("FAILURES")
    print("=" * 70)
    for name, e in failures.items():
      print(f"  {name}: {e}")


if __name__ == "__main__":
  print("=" * 70)
  print("SQTT InstOp Discovery from Tinygrad Kernels")
  print("=" * 70)
  print(f"Testing {len(TENSOR_TESTS)} tensor operations...\n")

  discovered, failures = discover_all_instops()
  print_summary(discovered, failures)
