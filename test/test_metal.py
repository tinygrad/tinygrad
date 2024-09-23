import unittest
from tinygrad.runtime.ops_metal import MetalDevice, MetalCompiler
from tinygrad.device import CompileError

class TestMetal(unittest.TestCase):
  def test_alloc_oom(self):
    device = MetalDevice("metal")
    with self.assertRaises(MemoryError):
      device.allocator.alloc(10000000000000000000)

  def test_compile_error(self):
    device = MetalDevice("metal")
    compiler = MetalCompiler(device)
    with self.assertRaises(CompileError):
      compiler.compile("this is not valid metal")
