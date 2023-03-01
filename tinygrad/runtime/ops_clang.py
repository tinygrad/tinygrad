import ctypes
import os, time
import numpy as np
import hashlib
import subprocess
from collections import defaultdict
from typing import Final, Dict
from tinygrad.helpers import prod
from tinygrad.ops import CompiledBuffer, RawBuffer
from tinygrad.codegen.gpu import GPUCodegen
import platform
OSX = platform.system() == "Darwin"

class RawMallocBuffer(RawBuffer):
  def __init__(self, size): self._buf = (ctypes.c_float * (size))()
  def copyin(self, b:np.ndarray): ctypes.memmove(self._buf, b.ctypes.data, b.size*4)
  def copyout(self, a:np.ndarray): np.copyto(a, np.ctypeslib.as_array(self._buf)[:a.size].reshape(a.shape))

class ClangProgram:
  kernel_cnt : Final[Dict[str, int]] = defaultdict(int)
  def __init__(self, name:str, prg:str):
    prg = "#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n" + prg
    # TODO: is there a way to not write this to disk?
    fn = f"/tmp/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{'dylib' if OSX else 'so'}"
    if not os.path.exists(fn):
      subprocess.check_output(['clang', '-shared', '-O2', '-Wall','-Werror', '-lm', '-fPIC', '-x', 'c', '-', '-o', fn+".tmp"], input=prg.encode('utf-8'))
      os.rename(fn+".tmp", fn)
    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name]
  def __call__(self, *args, wait=False):
    if wait: st = time.monotonic()
    self.fxn(*[x._buf for x in args[2:]])
    if wait: return time.monotonic()-st

class ClangBuffer(CompiledBuffer):
  @staticmethod
  def create_raw_buffer(shape): return RawMallocBuffer(4*prod(shape))
  @staticmethod
  def compile(ast, output_buffer):
    k = GPUCodegen(ast, output_buffer)
    return (k.codegen().build(ClangProgram), k.bufs, k.ret)
