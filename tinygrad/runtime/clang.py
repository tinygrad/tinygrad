import ctypes
import numpy as np
import subprocess
from tinygrad.ops import DEBUG

class CLBuffer:
  def __init__(self, size): self._cl = (ctypes.c_float * (size))()
  def copyin(self, b:np.ndarray): ctypes.memmove(self._cl, b.ctypes.data, b.size*4)
  def copyout(self, a:np.ndarray): np.copyto(a, np.ctypeslib.as_array(self._cl).reshape(a.shape))

class CLProgram:
  kernel_prefix, buffer_prefix, smem_prefix, barrier = "", "", "", ""
  gid = [f"gid[{i}]" for i in range(3)]
  lid = [f"lid[{i}]" for i in range(3)]
  extra_args = ['int gid[3]', 'int lid[3]']
  # TODO: remove name, factor out op_estimate and mem_estimate
  def __init__(self, name:str, prg:str, op_estimate=0, mem_estimate=0):
    if DEBUG >= 4: print(prg)  # TODO: outside runtime!
    lib = subprocess.check_output(['clang', '-x', 'c', '-'], input=prg.encode('utf-8'))
    print(lib)
