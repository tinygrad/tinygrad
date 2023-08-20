import unittest
import uuid
import hashlib
from tinygrad.runtime.lib import compile_cache, mktemp_clean

class TestCache(unittest.TestCase):
  compile_call_count = 0
  
  @compile_cache("test", str(uuid.uuid4()))
  def compile(self, name:str, prg:str, binary:bool=False, extension:str=""):
    self.compile_call_count += 1
    return hashlib.sha256(prg.encode()).digest()
  
  def test_compile_cache(self):
    prg1 = "example prg1 contents"
    prg2 = "example prg2 contents"
    cold_compile_res = self.compile("program", prg1)
    warm_compile_res = self.compile("program", prg1)
    assert len(cold_compile_res) > 0
    assert cold_compile_res == warm_compile_res
    assert self.compile_call_count == 1

    prg2_res = self.compile("program", prg2)
    self.compile("program", prg2)
    assert len(prg2_res) > 0
    assert self.compile_call_count == 2