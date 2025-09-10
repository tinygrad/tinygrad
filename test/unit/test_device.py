#!/usr/bin/env python
import unittest, os, subprocess, sys
from tinygrad import Tensor
from tinygrad.device import Device, Compiler
from tinygrad.helpers import diskcache_get, diskcache_put, getenv, Context, WIN

class TestDevice(unittest.TestCase):
  def test_canonicalize(self):
    self.assertEqual(Device.canonicalize(None), Device.DEFAULT)
    self.assertEqual(Device.canonicalize("CPU"), "CPU")
    self.assertEqual(Device.canonicalize("cpu"), "CPU")
    self.assertEqual(Device.canonicalize("GPU"), "GPU")
    self.assertEqual(Device.canonicalize("GPU:0"), "GPU")
    self.assertEqual(Device.canonicalize("gpu:0"), "GPU")
    self.assertEqual(Device.canonicalize("GPU:1"), "GPU:1")
    self.assertEqual(Device.canonicalize("gpu:1"), "GPU:1")
    self.assertEqual(Device.canonicalize("GPU:2"), "GPU:2")
    self.assertEqual(Device.canonicalize("disk:/dev/shm/test"), "DISK:/dev/shm/test")
    self.assertEqual(Device.canonicalize("disk:000.txt"), "DISK:000.txt")

  def test_getitem_not_exist(self):
    with self.assertRaises(ModuleNotFoundError):
      Device["TYPO"]

  def test_lowercase_canonicalizes(self):
    device = Device.DEFAULT
    Device.DEFAULT = device.lower()
    self.assertEqual(Device.canonicalize(None), device)
    Device.DEFAULT = device

  def test_env_overwrite_default_compiler(self):
    expect_failure = "\ntry: assert Device[Device.DEFAULT].compiler is None;\nexcept RuntimeError: pass"

    if Device.DEFAULT == "CPU" and not WIN:
      from tinygrad.runtime.support.compiler_cpu import CPULLVMCompiler, ClangJITCompiler
      try: _, _ = CPULLVMCompiler(), ClangJITCompiler()
      except Exception as e: self.skipTest(f"skipping compiler test: not all compilers: {e}")

      imports = "from tinygrad import Device; from tinygrad.runtime.support.compiler_cpu import CPULLVMCompiler, ClangJITCompiler"
      subprocess.run([f'DEV=CPU CPU_LLVM=1 python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, CPULLVMCompiler)"'],
                        shell=True, check=True)
      subprocess.run([f'DEV=CPU CPU_LLVM=0 python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, ClangJITCompiler)"'],
                        shell=True, check=True)
      subprocess.run([f'DEV=CPU CPU_CLANGJIT=0 CPU_LLVM=0 python3 -c "{imports}; {expect_failure}"'],
                        shell=True, check=True)
      subprocess.run([f'DEV=CPU CPU_CLANGJIT=0 python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, CPULLVMCompiler)"'],
                        shell=True, check=True)
      subprocess.run([f'DEV=CPU CPU_CLANGJIT=1 CPU_LLVM=1 python3 -c "{imports}; {expect_failure}"'],
                        shell=True, check=True)
    elif Device.DEFAULT == "AMD":
      from tinygrad.runtime.support.compiler_amd import HIPCompiler, AMDLLVMCompiler
      try: _, _ = HIPCompiler(Device[Device.DEFAULT].arch), AMDLLVMCompiler(Device[Device.DEFAULT].arch)
      except Exception as e: self.skipTest(f"skipping compiler test: not all compilers: {e}")

      imports = "from tinygrad import Device; from tinygrad.runtime.support.compiler_amd import HIPCompiler, AMDLLVMCompiler"
      subprocess.run([f'DEV=AMD AMD_LLVM=1 python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, AMDLLVMCompiler)"'],
                        shell=True, check=True)
      subprocess.run([f'DEV=AMD AMD_LLVM=0 python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, HIPCompiler)"'],
                        shell=True, check=True)
      subprocess.run([f'DEV=AMD AMD_HIP=1 python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, HIPCompiler)"'],
                        shell=True, check=True)
      subprocess.run([f'DEV=AMD AMD_HIP=1 AMD_LLVM=1 python3 -c "{imports}; {expect_failure}"'],
                        shell=True, check=True)
    else: self.skipTest("only run on CPU/AMD")

class MockCompiler(Compiler):
  def __init__(self, key): super().__init__(key)
  def compile(self, src) -> bytes: return src.encode()

class TestCompiler(unittest.TestCase):
  def test_compile_cached(self):
    diskcache_put("key", "123", None) # clear cache
    getenv.cache_clear()
    with Context(DISABLE_COMPILER_CACHE=0):
      self.assertEqual(MockCompiler("key").compile_cached("123"), str.encode("123"))
      self.assertEqual(diskcache_get("key", "123"), str.encode("123"))

  def test_compile_cached_disabled(self):
    diskcache_put("disabled_key", "123", None) # clear cache
    getenv.cache_clear()
    with Context(DISABLE_COMPILER_CACHE=1):
      self.assertEqual(MockCompiler("disabled_key").compile_cached("123"), str.encode("123"))
      self.assertIsNone(diskcache_get("disabled_key", "123"))

  def test_device_compile(self):
    getenv.cache_clear()
    with Context(DISABLE_COMPILER_CACHE=1):
      a = Tensor([0.,1.], device=Device.DEFAULT).realize()
      (a + 1).realize()

class TestRunAsModule(unittest.TestCase):
  def test_module_runs(self):
    p = subprocess.run([sys.executable, "-m", "tinygrad.device"],stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      env={**os.environ, "DEBUG": "1"}, timeout=10,)
    out = (p.stdout + p.stderr).decode()
    self.assertEqual(p.returncode, 0, msg=out)
    self.assertIn("CPU", out) # for sanity check

if __name__ == "__main__":
  unittest.main()
