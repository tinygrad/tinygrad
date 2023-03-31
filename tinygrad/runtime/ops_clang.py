import os, time, ctypes, hashlib, subprocess, platform
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

class ClangProgram:
  def __init__(self, name:str, prg:str):
    # this might not even be needed
    prg = "#![crate_type = \"dylib\"]\n" + prg

    # TODO: is there a way to not write this to disk?
    fn = f"/tmp/rustc_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{'dylib' if platform.system() == 'Darwin' else 'so'}"

    if not os.path.exists(fn):
      subprocess.check_output(['rustc', '--crate-type=dylib', '-Copt-level=3', '-o', fn+".tmp", "-"], input=prg.encode('utf-8'))
      os.rename(fn+".tmp", fn)
    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name+"_c"]

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait: st = time.monotonic()
    self.fxn(*[x._buf for x in args])
    if wait: return time.monotonic()-st

class ClangCodegen(CStyleCodegen):
  # TODO could add `static` as kernel_prefix, but we can't use -shared and DLL to test the code in __init__
  lang = CStyleLanguage(buffer_suffix="", kernel_prefix="#[inline(always)]\n")

ClangBuffer = Compiled(RawMallocBuffer, ClangCodegen, ClangProgram)
