import os, time, ctypes, hashlib, subprocess, platform
from tinygrad.helpers import DEBUG
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

class ClangProgram:
  def __init__(self, name:str, prg:str, binary:bool=False):
    fn = f"/tmp/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{'dylib' if platform.system() == 'Darwin' else 'so'}"
    if not binary:
      prg = "#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half __fp16\n#define uchar unsigned char\n#define bool uchar\n" + prg
      # TODO: is there a way to not write this to disk?
      # NOTE: --rtlib=compiler-rt fixes float16 on Linux, it defines __gnu_h2f_ieee and __gnu_f2h_ieee
      if not os.path.exists(fn):
        subprocess.check_output(['gcc', '-shared', '-O2', '-Wall','-Werror', '-lm', '-fPIC', '-x', 'c', '-', '-o', fn+".tmp"], input=prg.encode('utf-8'))
        os.rename(fn+".tmp", fn)
    else:
      if DEBUG >= 5: print(prg)
      # with open('kernel.s', 'w+') as f: f.write(prg)
      print(subprocess.run(["as", "-o", "kernel.o"], input=prg.encode('utf-8')))
      print(subprocess.run(["ld", "-shared", "-lm", "-fPIC", "kernel.o", "-o", fn]))
    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name]

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait: st = time.monotonic()
    out = self.fxn(*[x._buf for x in args])
    print("out", out)
    if wait: return time.monotonic()-st

class ClangCodegen(CStyleCodegen):
  lang = CStyleLanguage(buffer_suffix=" restrict")
  supports_float4: bool = False

ClangBuffer = Compiled(RawMallocBuffer, ClangCodegen, ClangProgram)
