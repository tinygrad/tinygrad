#!/usr/bin/env python
import numpy as np
import unittest
import secrets
import string
from tinygrad.tensor import Tensor
from tinygrad.ops import Device
from tinygrad.helpers import cache_filepath
import tinygrad.runtime.ops_clang

def generate_random_string(length=16):
  alphabet = string.ascii_letters + string.digits
  return ''.join(secrets.choice(alphabet) for _ in range(length))

def check_compile_folder(directory):
  return [file for file in directory.iterdir() if file.is_file()]

class TestKernelCache(unittest.TestCase):
  def test_kernel_cache(self):
    if Device.DEFAULT not in ["CLANG"]:
      self.skipTest("No custom kernel cache is implemented")

    orig_toolchain_hash = tinygrad.runtime.ops_clang.TOOLCHAIN_HASH
    tinygrad.runtime.ops_clang.TOOLCHAIN_HASH = "ci-"+generate_random_string()
    cache_dir = cache_filepath(f"clang-{tinygrad.runtime.ops_clang.TOOLCHAIN_HASH}", "").parent
    assert len(check_compile_folder(cache_dir)) == 0, "should be empty"

    a = Tensor.rand(4,4)
    b = Tensor.rand(4,4)
    x = a + b
    x.realize()

    assert len(check_compile_folder(cache_dir)) == 1, "kernel is not cached"

    orig_compile_func = tinygrad.runtime.ops_clang.ClangBuffer.runtime.compile
    tinygrad.runtime.ops_clang.ClangBuffer.runtime.compile = None # making it not callable

    a1 = Tensor.rand(4,4)
    b1 = Tensor.rand(4,4)
    x1 = a1 + b1
    x1.realize() # Same kernel should be from cache.

    tinygrad.runtime.ops_clang.ClangBuffer.runtime.compile = orig_compile_func
    tinygrad.runtime.ops_clang.TOOLCHAIN_HASH = orig_toolchain_hash

if __name__ == "__main__":
  unittest.main()
