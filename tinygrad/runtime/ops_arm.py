import numpy as np
import operator
import os
import platform
import tempfile
import subprocess
import ctypes
import hashlib
from typing import Callable, Dict, Tuple, Optional
from tinygrad.helpers import dtypes, DType
from tinygrad.ops import Compiled 
from tinygrad.codegen.assembly_arm import ARMCodegen
from tinygrad.runtime.lib import RawMallocBuffer


class ARMProgram:
  def __init__(self, name:str, prg:str, **args):
    print(prg)
#     prg = f"""
# .arch armv8-a
# .text
# .global _{name} 
# .balign 4
# _{name}:
#     ldr s0, [x1]
#     ldr s1, [x2]
#     fadd s0, s0, s1
#     str s0, [x0]
#     ret
# """
    fn = f"/tmp/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{'dylib' if platform.system() == 'Darwin' else 'so'}"
    print(subprocess.run(["as","-arch", "arm64", "-o", "{name}.o"], input=prg.encode('utf-8')))
    print(subprocess.run(["clang", "-shared","{name}.o", "-o", fn]))

    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name] 

  def __call__(self, global_size, local_size, *args, wait=False):
    self.fxn.argtypes = [ctypes.c_uint64 for i in range(len(args))]
    self.fxn.restype = ctypes.c_uint64
    out = self.fxn(*[ctypes.addressof(x._buf) for x in args])
    print("out", out)
    if wait: return time.monotonic()-st
    return out 
 

ARMBuffer = Compiled(RawMallocBuffer, ARMCodegen, ARMProgram)
