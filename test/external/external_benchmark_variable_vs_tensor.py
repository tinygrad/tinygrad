"""Compare symbolic scalar arguments with pre-realized size-one Tensor arguments.

This gives the Tensor path its best case: buffer allocation and upload happen before
measurement. Run on a selected backend with, for example:

  DEV=CPU python test/external/external_benchmark_variable_vs_tensor.py
  DEV=CL SIZES=1,1024,1048576 SAMPLES=7 ITERS=50 python test/external/external_benchmark_variable_vs_tensor.py
"""
from __future__ import annotations

import contextlib, io, os, platform, statistics, sys, time
from typing import Callable, Sequence

from tinygrad import Context, Device, Tensor, TinyJit, Variable, dtypes
from tinygrad.helpers import GlobalCounters
from tinygrad.uop.ops import Ops, UOp

WARMUPS = int(os.getenv("WARMUPS", "3"))
SAMPLES = int(os.getenv("SAMPLES", "7"))
ITERS = int(os.getenv("ITERS", "50"))
SIZES = tuple(int(x) for x in os.getenv("SIZES", "1,1024,1048576").split(","))
VALUES = tuple(range(1, 9))


def _program(jit) -> UOp:
  assert jit.captured is not None
  return next(u for u in jit.captured.linear.toposort() if u.op is Ops.PROGRAM)


def _measure(device:str, fxn:Callable, x:Tensor, values:Sequence[UOp|Tensor]) -> tuple[float, float, float]:
  submit_samples, wait_samples, kernel_samples = [], [], []
  for _ in range(SAMPLES):
    Device[device].synchronize()
    st = time.perf_counter_ns()
    for i in range(ITERS): fxn(x, values[i % len(values)])
    submit_samples.append((time.perf_counter_ns() - st) / ITERS / 1e3)
    Device[device].synchronize()

    GlobalCounters.reset()
    with Context(DEBUG=2), contextlib.redirect_stdout(io.StringIO()):
      st = time.perf_counter_ns()
      for i in range(ITERS): fxn(x, values[i % len(values)])
      wait_samples.append((time.perf_counter_ns() - st) / ITERS / 1e3)
    assert GlobalCounters.kernel_count == ITERS, f"expected one kernel per iteration, got {GlobalCounters.kernel_count}/{ITERS}"
    kernel_samples.append(GlobalCounters.time_sum_s / GlobalCounters.kernel_count * 1e6)
  return statistics.median(submit_samples), statistics.median(wait_samples), statistics.median(kernel_samples)


def _benchmark_size(device:str, size:int) -> dict[str, tuple[float, float, float]]:
  x = Tensor.arange(size, dtype=dtypes.int32).clone(device).realize()

  @TinyJit
  def scalar_arg(x:Tensor, value:UOp) -> Tensor: return (x + Tensor(value)).realize()

  @TinyJit
  def tensor_arg(x:Tensor, value:Tensor) -> Tensor: return (x + value).realize()

  scalar_values = [Variable("value", 0, 100, dtype=dtypes.int32).bind(value) for value in VALUES]
  tensor_values = [Tensor([value], dtype=dtypes.int32, device=device).clone().realize() for value in VALUES]
  for i in range(WARMUPS):
    scalar_out, tensor_out = scalar_arg(x, scalar_values[i % len(VALUES)]), tensor_arg(x, tensor_values[i % len(VALUES)])
  Device[device].synchronize()

  expected = (VALUES[(WARMUPS - 1) % len(VALUES)], size - 1 + VALUES[(WARMUPS - 1) % len(VALUES)])
  assert (scalar_out[0].item(), scalar_out[-1].item()) == expected
  assert (tensor_out[0].item(), tensor_out[-1].item()) == expected

  scalar_program, tensor_program = _program(scalar_arg), _program(tensor_arg)
  scalar_vars = [v for v in scalar_program.arg.vars if v.expr not in scalar_program.arg.runtimevars]
  tensor_vars = [v for v in tensor_program.arg.vars if v.expr not in tensor_program.arg.runtimevars]
  assert [v.expr for v in scalar_vars] == ["value"] and tensor_vars == []
  assert len(tensor_program.arg.globals) == len(scalar_program.arg.globals) + 1

  return {
    "scalar": _measure(device, scalar_arg, x, scalar_values),
    "tensor": _measure(device, tensor_arg, x, tensor_values),
  }


def main():
  assert WARMUPS >= 2, "TinyJit needs one warmup and one capture call"
  assert SAMPLES > 0 and ITERS > 0 and all(size > 0 for size in SIZES)
  device = Device.DEFAULT
  print(f"device={device} runtime={type(Device[device]).__name__} python={sys.version.split()[0]} platform={platform.platform()}")
  print(f"warmups={WARMUPS} samples={SAMPLES} iterations={ITERS} sizes={SIZES}")
  print("Tensor values are pre-realized; allocation and host-to-device upload are excluded.")
  print("size       path       submit_us      wait_us    kernel_us")
  for size in SIZES:
    results = _benchmark_size(device, size)
    for path, values in results.items(): print(f"{size:<10d} {path:<8s} {values[0]:>12.3f} {values[1]:>12.3f} {values[2]:>12.3f}")
    ratios = tuple(results["tensor"][i] / results["scalar"][i] for i in range(3))
    print(f"{size:<10d} ratio    {ratios[0]:>12.3f} {ratios[1]:>12.3f} {ratios[2]:>12.3f}  (tensor/scalar)")


if __name__ == "__main__": main()
