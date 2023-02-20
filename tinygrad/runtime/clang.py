import ctypes
import os
import numpy as np
import hashlib
import subprocess
from typing import List
from tinygrad.ops import DEBUG
import platform
OSX = platform.system() == "Darwin"

class CLBuffer:
  def __init__(self, size): self._cl = (ctypes.c_float * (size))()
  def copyin(self, b:np.ndarray): ctypes.memmove(self._cl, b.ctypes.data, b.size*4)
  def copyout(self, a:np.ndarray):
    np.copyto(a, np.ctypeslib.as_array(self._cl)[:a.size].reshape(a.shape))

class CLProgram:
  kernel_prefix, buffer_prefix, smem_prefix, barrier = "", "", "", ""
  gid = [f"gid[{i}]" for i in range(3)]
  lid = [f"lid[{i}]" for i in range(3)]
  extra_args : List[str] = []
  # TODO: remove name, factor out op_estimate and mem_estimate
  def __init__(self, name:str, prg:str, op_estimate=0, mem_estimate=0):
    self.name, self.prg = name, prg
    prg = "#include <math.h>\n#define max(x,y) fmax(x,y)\n" + prg
    if DEBUG >= 4: print(prg)  # TODO: outside runtime!
    # TODO: is there a way to not write this to disk?
    fn = f"/tmp/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{'dylib' if OSX else 'so'}"
    if not os.path.exists(fn):
      subprocess.check_output(['clang', '-shared', '-O2', '-x', 'c', '-', '-o', fn+".tmp"], input=prg.encode('utf-8'))
      os.rename(fn+".tmp", fn)
    self.fxn = ctypes.CDLL(fn)[name]
  def __call__(self, *args): self.fxn(*args[2:])
