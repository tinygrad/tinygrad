from tinygrad import Device, dtypes
from tinygrad.helpers import getenv, colorize_float
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from test.external.fuzz_linearizer import get_fuzz_rawbufs
from tinygrad.codegen.heuristic import hand_coded_optimizations
from tinygrad.engine.search import bufs_from_lin
from tinygrad.engine.realize import CompiledRunner
from tinygrad.tensor import _to_np_dtype
from tinygrad.runtime.ops_amd import AMDDevice
from contextlib import contextmanager
import numpy as np

am_signal_pages, am_signal_pool, am_devices = [], [], []
amd_signal_pages, amd_signal_pool, amd_devices = [], [], []

@contextmanager
def run_amd():
  global amd_signal_pages, amd_signal_pool, amd_devices
  AMDDevice.driverless = False
  AMDDevice.signal_pages, AMDDevice.signal_pool, AMDDevice.devices = amd_signal_pages, amd_signal_pool, amd_devices
  yield
  amd_signal_pages, amd_signal_pool, amd_devices = AMDDevice.signal_pages, AMDDevice.signal_pool, AMDDevice.devices
  AMDDevice.signal_pages, AMDDevice.signal_pool, AMDDevice.devices = [], [], []

@contextmanager
def run_am():
  global am_signal_pages, am_signal_pool, am_devices
  AMDDevice.driverless = True
  AMDDevice.signal_pages, AMDDevice.signal_pool, AMDDevice.devices = am_signal_pages, am_signal_pool, am_devices
  yield
  am_signal_pages, am_signal_pool, am_devices = AMDDevice.signal_pages, AMDDevice.signal_pool, AMDDevice.devices
  AMDDevice.signal_pages, AMDDevice.signal_pool, AMDDevice.devices = [], [], []

if __name__ == "__main__":
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)

  assert not AMDDevice.driverless, "No amdgpu module found to stress against"
  with run_amd():
    amddev = Device["AMD"]

  with run_am():
    amdev = Device["AMD:1"]

  single = getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]

  average_tm_cuda, average_tm_nv = 0, 0
  for num,ast in enumerate(ast_strs):
    with run_amd():
      amdlin = ast_str_to_lin(ast, opts=amddev.renderer)
      amdlin = hand_coded_optimizations(amdlin)
      has_bf16 = any(b.dtype == dtypes.bfloat16 for b in amdlin.membufs)

      cuda_prg = CompiledRunner(amdlin.to_program())
      amdbufs = bufs_from_lin(amdlin)
      test_amdbufs = get_fuzz_rawbufs(amdlin) if not has_bf16 else amdbufs

    with run_am():
      amlin = ast_str_to_lin(ast, opts=amdev.renderer)
      nv_prg = CompiledRunner(amlin.to_program())
      ambufs = bufs_from_lin(amlin)
      test_ambufs = get_fuzz_rawbufs(amlin) if not has_bf16 else ambufs
      if not has_bf16:
        for i,rawbuf in enumerate(test_ambufs): rawbuf.copyin(test_amdbufs[i].as_buffer())

    # warmup
    tm_cuda, tm_nv, failed = [], [], False
    with run_amd():
      try:
        cuda_prg(test_amdbufs, {}, wait=True)
        for i in range(5): tm_cuda.append(cuda_prg(amdbufs, {}, wait=True))
      except RuntimeError:
        print("AMD FAILED")
        tm_cuda = [1e9]
        failed = True

    with run_am():
      try:
        AMDDevice.driverless = True
        nv_prg(test_ambufs, {}, wait=True)
        for i in range(5): tm_nv.append(nv_prg(ambufs, {}, wait=True))
      except RuntimeError:
        print("NV FAILED")
        tm_nv = [1e9]
        failed = True

    if not failed and not has_bf16:
      with run_amd():
        curesult = np.frombuffer(test_amdbufs[0].as_buffer(), _to_np_dtype(test_amdbufs[0].dtype))

      with run_am():
        nvresult = np.frombuffer(test_ambufs[0].as_buffer(), _to_np_dtype(test_ambufs[0].dtype))

      np.testing.assert_allclose(curesult, nvresult, rtol=1e-2, atol=1e-2)

    average_tm_cuda += min(tm_cuda)
    average_tm_nv += min(tm_nv)
    ratio = min(tm_nv)/min(tm_cuda)
    print(f"{average_tm_nv/average_tm_cuda:5.2f}x -- {num:4d} {colorize_float(ratio)}  {min(tm_nv)*1e6:7.2f} us", amlin.name)
    if ratio > 1.04: print(f"NV slower {ratio}", amlin.ast, amlin.applied_opts)
