import os, time, ctypes, hashlib, subprocess, platform, tempfile
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

class DynLib:
  def __init__(self, sys:str):
    self.defs = '#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half __fp16\n#define uchar unsigned char\n#define bool uchar\n'
    self.args = {
      'Windows': {'cflags':'', 'ext':'dll', 'exp':'__declspec(dllexport)'},
      'Linux': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'so', 'exp':''},
      'Darwin': {'cflags':'', 'ext':'dylib', 'exp':''}
    }[sys]
    self.ext = self.args['ext']
    self.exp = self.args['exp']

  def cc(self, fn:str):
    return ('clang -shared -O2 -Wall -Werror -x c '+self.args['cflags']+' - -o '+fn).split()

  def src(self, prg:str):
    return (self.defs + prg).encode('utf-8')

ClangDll = DynLib(platform.system())

class ClangProgram:
  def __init__(self, name:str, prg:str):
    # TODO: is there a way to not write this to disk?
    fn = f"{tempfile.gettempdir()}/clang_{hashlib.md5(ClangDll.src(prg)).hexdigest()}.{ClangDll.ext}"
    if not os.path.exists(fn):
      subprocess.check_output(args=ClangDll.cc(fn+'.tmp'), input=ClangDll.src(prg))
      os.rename(fn+'.tmp', fn)
    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name]

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait: st = time.monotonic()
    self.fxn(*[x._buf for x in args])
    if wait: return time.monotonic()-st

class ClangCodegen(CStyleCodegen):
  lang = CStyleLanguage(kernel_prefix=ClangDll.exp, buffer_suffix=" restrict")
  supports_float4: bool = False

ClangBuffer = Compiled(RawMallocBuffer, ClangCodegen, ClangProgram)
