import os, time, ctypes, hashlib, subprocess, platform
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

class ClangProgram:
  def __init__(self, name:str, prg:str):
    prg = "#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define half __fp16\n" + prg
    # TODO: is there a way to not write this to disk?
    fn = f"/tmp/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{'dylib' if platform.system() == 'Darwin' else 'so'}"
    # TODO call rustc to produce a shared lib with the same names as one would expect from clang
    # https://doc.rust-lang.org/reference/linkage.html

    # NOTE: --rtlib=compiler-rt fixes float16 on Linux, it defines __gnu_h2f_ieee and __gnu_f2h_ieee
    # if not os.path.exists(fn):
      # subprocess.check_output(['clang', '-shared', '-O2', '-Wall','-Werror', '-lm', '--rtlib=compiler-rt', '-fPIC', '-std=c2x', '-x', 'c', '-', '-o', fn+".tmp"], input=prg.encode('utf-8'))
      # os.rename(fn+".tmp", fn)
    # self.lib = ctypes.CDLL(fn)
    # self.fxn = self.lib[name]

  def __call__(self, global_size, local_size, *args, wait=False):
    pass
    # if wait: st = time.monotonic()
    # self.fxn(*[x._buf for x in args])
    # if wait: return time.monotonic()-st

class ClangCodegen(CStyleCodegen):
  # TODO could add `static` as kernel_prefix, but we can't use -shared and DLL to test the code in __init__
  lang = CStyleLanguage(buffer_suffix="", kernel_prefix="#[inline(always)]\n")
  supports_float4: bool = False

ClangBuffer = Compiled(RawMallocBuffer, ClangCodegen, ClangProgram)
