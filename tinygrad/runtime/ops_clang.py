import os, time, ctypes, hashlib, subprocess, platform, tempfile
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

class Config:
  program = '#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half __fp16\n#define uchar unsigned char\n#define bool uchar\n'
  platform = {
    'Windows': {'cflags':'', 'ext':'dll', 'export':'__declspec(dllexport)'},
    'Linux': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'so', 'export':'__attribute__((visibility("default")))'},
    'Darwin': {'cflags':'', 'ext':'dylib', 'export':''}
  }[platform.system()]
  command = 'clang -shared -O2 -Wall -Werror -x c '+platform['cflags']+' - -o '
  extension = platform['ext']
  export = platform['export']

class ClangProgram:
  def __init__(self, name:str, prg:str):
    prg = Config.program + prg
    # TODO: is there a way to not write this to disk?
    fn = f"{tempfile.gettempdir()}/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{Config.extension}"
    if not os.path.exists(fn):
      subprocess.check_output((Config.command + fn+'.tmp').split(), input=prg.encode('utf-8'))
      os.rename(fn+".tmp", fn)
    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name]

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait: st = time.monotonic()
    self.fxn(*[x._buf for x in args])
    if wait: return time.monotonic()-st

class ClangCodegen(CStyleCodegen):
  lang = CStyleLanguage(kernel_prefix=Config.export, buffer_suffix=" restrict")
  supports_float4: bool = False

ClangBuffer = Compiled(RawMallocBuffer, ClangCodegen, ClangProgram)
