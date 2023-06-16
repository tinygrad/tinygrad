import time, ctypes, hashlib, subprocess, platform
from tinygrad.codegen.assembly_x86 import X86Codegen
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawMallocBuffer

class X86Program:
  def __init__(self, name:str, prg:str, binary:bool=False):
    fn = f"/tmp/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{'dylib' if platform.system() == 'Darwin' else 'so'}"
    print(prg)
    # with open('kernel.s', 'w+') as f: f.write(prg)
    print(subprocess.run(["as", "-o", "kernel.o"], input=prg.encode('utf-8')))
    print(subprocess.run(["ld", "-shared", "kernel.o", "-o", fn]))

    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name]

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait: st = time.monotonic()
    self.fxn.argtypes = [ctypes.c_uint64 for i in range(len(args))]
    self.fxn.restype = ctypes.c_uint64
    out = self.fxn(*[ctypes.addressof(x._buf) for x in args])
    print("out", out)
    if wait: return time.monotonic()-st

X86Buffer = Compiled(RawMallocBuffer, X86Codegen, X86Program)