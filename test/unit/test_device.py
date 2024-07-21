#!/usr/bin/env python
import unittest
from unittest.mock import patch
import os
from tinygrad.device import Device, Compiler
from tinygrad.helpers import diskcache_get, diskcache_put, getenv

class TestDevice(unittest.TestCase):
  def test_canonicalize(self):
    assert Device.canonicalize(None) == Device.DEFAULT
    assert Device.canonicalize("CPU") == "CPU"
    assert Device.canonicalize("cpu") == "CPU"
    assert Device.canonicalize("GPU") == "GPU"
    assert Device.canonicalize("GPU:0") == "GPU"
    assert Device.canonicalize("gpu:0") == "GPU"
    assert Device.canonicalize("GPU:1") == "GPU:1"
    assert Device.canonicalize("gpu:1") == "GPU:1"
    assert Device.canonicalize("GPU:2") == "GPU:2"
    assert Device.canonicalize("disk:/dev/shm/test") == "DISK:/dev/shm/test"

class MockCompiler(Compiler):
  def __init__(self, key): super().__init__(key)
  def compile(self, src, fname) -> bytes: return src.encode()

class TestCompiler(unittest.TestCase):
  def test_compile_cached(self):
    diskcache_put("key", {'src': "123", 'fname': ''}, None) # clear cache
    getenv.cache_clear()
    with patch.dict(os.environ, {"DISABLE_COMPILER_CACHE": "0"}, clear=True):
      assert MockCompiler("key").compile_cached("123", None) == str.encode("123")
      assert diskcache_get("key", {'src': "123", 'fname':''}) == str.encode("123")

  def test_compile_cached_disabled(self):
    diskcache_put("disabled_key", {'src': "123", 'fname': ''}, None) # clear cache
    getenv.cache_clear()
    with patch.dict(os.environ, {"DISABLE_COMPILER_CACHE": "1"}, clear=True):
      assert MockCompiler("disabled_key").compile_cached("123", None) == str.encode("123")
      assert diskcache_get("disabled_key", {'src': "123", 'fname': ''}) is None

if __name__ == "__main__":
  unittest.main()
