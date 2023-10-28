#!/usr/bin/env python
import unittest
import secrets
import string
import tempfile
import pathlib
from tinygrad.tensor import Tensor
from tinygrad.ops import Device
from tinygrad.helpers import cache_compiled
import tinygrad.runtime.ops_clang

def generate_random_string(length=16):
  alphabet = string.ascii_letters + string.digits
  return ''.join(secrets.choice(alphabet) for _ in range(length))

class TestKernelCache(unittest.TestCase):
  compile_call_count = 0

  @cache_compiled
  def __helper_test_compile(self, prg, output_file=pathlib.Path(tempfile.mktemp()), **kwargs):
    self.compile_call_count += 1
    return prg.encode()

  def test_compile_cache(self):
    prg1 = generate_random_string(64) + "a"
    prg2 = generate_random_string(64) + "b"
    cold_compile_res = self.__helper_test_compile(prg1)
    warm_compile_res = self.__helper_test_compile(prg1)
    assert cold_compile_res == warm_compile_res == prg1.encode()
    assert self.compile_call_count == 1

    prg2_res = self.__helper_test_compile(prg2)
    assert prg2_res == prg2.encode()
    assert self.compile_call_count == 2

  def test_kernel_cache_in_action(self):
    if Device.DEFAULT not in ["CLANG"]:
      self.skipTest("No custom kernel cache is implemented")

    a = Tensor.rand(4,4)
    b = Tensor.rand(4,4)
    x = a + b
    x.realize()

    orig_compile_func = tinygrad.runtime.ops_clang.ClangBuffer.runtime.compile
    tinygrad.runtime.ops_clang.ClangBuffer.runtime.compile = None # making it not callable

    a1 = Tensor.rand(4,4)
    b1 = Tensor.rand(4,4)
    x1 = a1 + b1
    x1.realize() # Same kernel should be from cache.

    tinygrad.runtime.ops_clang.ClangBuffer.runtime.compile = orig_compile_func

if __name__ == "__main__":
  unittest.main()
