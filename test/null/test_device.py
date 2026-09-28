#!/usr/bin/env python
import unittest, os, subprocess
from tinygrad import Tensor
from tinygrad.device import Device, Compiler
from tinygrad.helpers import diskcache_get, diskcache_put, getenv, Context, Target, DEV

class TestDevice(unittest.TestCase):
  def test_canonicalize(self):
    self.assertEqual(Device.canonicalize(None), Device.DEFAULT)
    self.assertEqual(Device.canonicalize("CPU"), "CPU")
    self.assertEqual(Device.canonicalize("cpu"), "CPU")
    self.assertEqual(Device.canonicalize("CL"), "CL")
    self.assertEqual(Device.canonicalize("CL:0"), "CL")
    self.assertEqual(Device.canonicalize("cl:0"), "CL")
    self.assertEqual(Device.canonicalize("CL:1"), "CL:1")
    self.assertEqual(Device.canonicalize("cl:1"), "CL:1")
    self.assertEqual(Device.canonicalize("CL:2"), "CL:2")
    self.assertEqual(Device.canonicalize("disk:/dev/shm/test"), "DISK:/dev/shm/test")
    self.assertEqual(Device.canonicalize("disk:000.txt"), "DISK:000.txt")

  def test_getitem_not_exist(self):
    with self.assertRaises(ModuleNotFoundError):
      Device["TYPO"]

  def test_lowercase_canonicalizes(self):
    device = Device.DEFAULT
    with Context(DEV=device.lower()):
      self.assertEqual(Device.canonicalize(None), device)

  def test_set_device_default_raises(self):
    with self.assertRaisesRegex(AttributeError, "setting Device.DEFAULT is deprecated"):
      Device.DEFAULT = "CPU"

  def test_old_device_env_raises(self):
    result = subprocess.run(['python3', '-c', 'from tinygrad import Device; Device.DEFAULT'],
                            env={**os.environ, "CPU": "1", "DEV": ""}, capture_output=True)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(b"deprecated", result.stderr)

  def test_dev_contextvar(self):
    orig_dev = Device.DEFAULT
    with Context(DEV="PYTHON"): self.assertEqual(Tensor.empty(1).device, "PYTHON")
    with Context(DEV="NULL"): self.assertEqual(Tensor.empty(1).device, "NULL")
    self.assertEqual(Tensor.empty(1).device, orig_dev)

class TestDevVar(unittest.TestCase):
  def test_parse(self):
    for d, t in [("AMD", Target(device="AMD", renderer="")), ("AMD:LLVM", Target(device="AMD", renderer="LLVM")),
                 (":LLVM", Target(device="", renderer="LLVM")), ("AMD::gfx1100", Target(device="AMD", arch="gfx1100")),
                 ("AMD:LLVM:gfx1100", Target(device="AMD", renderer="LLVM", arch="gfx1100")), ("::gfx1100", Target(arch="gfx1100")),
                 ("USB+", Target(interface="USB")), ("USB+AMD", Target(device="AMD", interface="USB")),
                 ("PCI:0+AMD", Target(device="AMD", interface="PCI", indices="0")), (":0+AMD", Target(device="AMD", indices="0")),
                 ("PCI:0,1+AMD", Target(device="AMD", interface="PCI", indices="0,1")),
                 ("QCOM;USB+AMD", [Target(device="QCOM"), Target(device="AMD", interface="USB")])]:
      with Context(DEV=d):
        self.assertEqual(DEV.value, t if isinstance(t, list) else [t])
        self.assertEqual(str(DEV), d)

  def test_target(self):
    with Context(DEV="CPU"): self.assertEqual(DEV.target("CPU"), Target("CPU"))
    with Context(DEV="CPU:LLVM"): self.assertEqual(DEV.target("CPU"), Target("CPU", "LLVM"))
    with Context(DEV=":LLVM"): self.assertEqual(DEV.target("CPU"), Target("CPU", "LLVM"))
    with Context(DEV="AMD:LLVM"): self.assertEqual(DEV.target("CPU"), Target("CPU"))
    with Context(DEV=""): self.assertEqual(DEV.target("CPU"), Target("CPU"))
    with Context(DEV="QCOM:IR3;AMD:LLVM"):
      self.assertEqual(DEV.target("QCOM"), Target("QCOM", "IR3"))
      self.assertEqual(DEV.target("AMD"), Target("AMD", "LLVM"))
      self.assertEqual(DEV.target("CPU"), Target("CPU"))

  def test_dev_arch_override(self):
    with Context(DEV="NULL::gfx1100"):
      self.assertEqual(Device["NULL"].renderer.target.arch, "gfx1100")

class MockCompiler(Compiler):
  def __init__(self, key): super().__init__(key)
  def compile(self, src) -> bytes: return src.encode()

class TestCompiler(unittest.TestCase):
  def test_compile_cached(self):
    diskcache_put("key", "123", None) # clear cache
    getenv.cache_clear()
    with Context(CCACHE=1):
      self.assertEqual(MockCompiler("key").compile_cached("123"), str.encode("123"))
      self.assertEqual(diskcache_get("key", "123"), str.encode("123"))

  def test_compile_cached_disabled(self):
    diskcache_put("disabled_key", "123", None) # clear cache
    getenv.cache_clear()
    with Context(CCACHE=0):
      self.assertEqual(MockCompiler("disabled_key").compile_cached("123"), str.encode("123"))
      self.assertIsNone(diskcache_get("disabled_key", "123"))

  def test_device_compile(self):
    getenv.cache_clear()
    with Context(CCACHE=0):
      a = Tensor([0.,1.]).realize()
      (a + 1).realize()

if __name__ == "__main__":
  unittest.main()
