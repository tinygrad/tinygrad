#!/usr/bin/env python
import unittest
import secrets
import string
from tinygrad.tensor import Tensor
from tinygrad.ops import Device
from tinygrad.helpers import diskcache

def generate_random_string(length=16):
  alphabet = string.ascii_letters + string.digits
  return ''.join(secrets.choice(alphabet) for _ in range(length))

compile_call_count = 0

@diskcache
def helper_test_compile(prg:str) -> bytes:
  global compile_call_count
  compile_call_count += 1
  return prg.encode()

class TestKernelCache(unittest.TestCase):
  def test_compile_cache(self):
    prg1 = generate_random_string(64) + "a"
    prg2 = generate_random_string(64) + "b"
    cold_compile_res = helper_test_compile(prg1)
    warm_compile_res = helper_test_compile(prg1)
    assert cold_compile_res == warm_compile_res == prg1.encode()
    assert compile_call_count == 1

    prg2_res = helper_test_compile(prg2)
    assert prg2_res == prg2.encode()
    assert compile_call_count == 2

  def test_kernel_cache_in_action(self):
    if Device.DEFAULT not in ["CLANG"]:
      self.skipTest("No custom kernel cache is implemented")

    a = Tensor.rand(4,4)
    b = Tensor.rand(4,4)
    x = a + b
    x.realize()

    orig_compile_func = Device['CLANG'].compiler
    Device['CLANG'].compiler = None # making it not callable

    a1 = Tensor.rand(4,4)
    b1 = Tensor.rand(4,4)
    x1 = a1 + b1
    x1.realize() # Same kernel should be from cache.

    Device['CLANG'].compiler = orig_compile_func

if __name__ == "__main__":
  unittest.main()
