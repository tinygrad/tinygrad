#!/usr/bin/env python
import unittest, os, subprocess, sys
from unittest.mock import patch
from tinygrad.device import Device, enumerate_devices_str
from tinygrad.helpers import Context, WIN, OSX
from tinygrad.runtime.support.c import DLL

@unittest.skipIf(Device.DEFAULT != "CPU", "only run on CPU")
class TestDevice(unittest.TestCase):
  def test_nonexistent_renderer(self):
    with self.assertRaisesRegex(RuntimeError, "has no renderer"):
      with Context(DEV="CPU:TYPO"): Device[Device.DEFAULT].renderer
    with self.assertRaisesRegex(RuntimeError, "did you mean: 'CLANG'"):
      with Context(DEV="CPU:CLANGJIT"): Device[Device.DEFAULT].renderer

  def test_old_renderer_env_raises(self):
    result = subprocess.run(['python3', '-c', 'from tinygrad import Device; Device[Device.DEFAULT].renderer'],
                            env={**os.environ, "DEV": "CPU", "CPU_LLVM": "1"}, capture_output=True)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(b"deprecated, use DEV=CPU:LLVM instead", result.stderr)

  @unittest.skipIf(WIN, "skipping windows test") # TODO: subprocess causes memory violation?
  def test_env_overwrite_default_compiler(self):
    from tinygrad.runtime.support.compiler_cpu import ClangCompiler
    from tinygrad.runtime.support.compiler_llvm import CPULLVMCompiler
    try: _, _ = CPULLVMCompiler(arch:=Device["CPU"].renderer.target.arch.split(",")), ClangCompiler(arch)
    except Exception as e: self.skipTest(f"skipping compiler test: not all compilers: {e}")

    imports = ("from tinygrad import Device; from tinygrad.runtime.support.compiler_cpu import ClangCompiler; "
               "from tinygrad.runtime.support.compiler_llvm import CPULLVMCompiler")
    for device, compiler in [("CPU:LLVM", "CPULLVMCompiler"), ("CPU", "ClangCompiler"), ("CPU:CLANG", "ClangCompiler")]:
      subprocess.run([sys.executable, '-c', f"{imports}; assert isinstance(Device[Device.DEFAULT].compiler, {compiler})"],
                     check=True, env={**os.environ, "DEV": device})

  @unittest.skipIf(WIN, "skipping windows test")
  def test_env_online(self):
    from tinygrad.runtime.support.compiler_cpu import ClangCompiler
    from tinygrad.runtime.support.compiler_llvm import CPULLVMCompiler
    try: _, _ = CPULLVMCompiler(arch:=Device["CPU"].renderer.target.arch.split(",")), ClangCompiler(arch)
    except Exception as e: self.skipTest(f"skipping compiler test: not all compilers: {e}")

    with Context(DEV="CPU:LLVM"):
      inst = Device["CPU"].compiler
      self.assertIsInstance(Device["CPU"].compiler, CPULLVMCompiler)
    with Context(DEV="CPU"):
      self.assertIsInstance(Device["CPU"].compiler, ClangCompiler)
    with Context(DEV="CPU:LLVM"):
      self.assertIsInstance(Device["CPU"].compiler, CPULLVMCompiler)
      assert inst is Device["CPU"].compiler  # cached

  def test_compiler_autodetect_fallback(self):
    from tinygrad.runtime.support.compiler_llvm import CPULLVMCompiler

    try: CPULLVMCompiler(Device["CPU"].renderer.target.arch.split(","))
    except Exception as e: self.skipTest(f"skipping: LLVM not available: {e}")

    dev = Device["CPU"]
    with Context(DEV="CPU"), patch.dict(dev.cached_renderer, clear=True):
      with patch("tinygrad.renderer.cstyle.ClangRenderer.__init__", side_effect=RuntimeError("broken")):
        self.assertIsInstance(dev.renderer.compiler, CPULLVMCompiler)

@unittest.skip("this test is broken if you have tinymesa installed")
@unittest.skipIf(OSX and 'libclang' in DLL._loaded_, "MTLCompiler can't be loaded after libclang on OSX")
class TestRunAsModule(unittest.TestCase):
  def test_module_runs(self):
    cpu_line = [l for l in enumerate_devices_str() if "CPU" in l][0]
    self.assertIn("PASS", cpu_line, f"expected CPU to PASS, got: {cpu_line}")

if __name__ == "__main__":
  unittest.main()
