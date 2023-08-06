import os, time, ctypes, hashlib, subprocess, platform, tempfile, functools
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.linearizer import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage

args = {
  'Windows': {'cflags':'', 'ext':'dll', 'exp':'__declspec(dllexport)'},
  'Linux': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'so', 'exp':''},
  'Darwin': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'dylib', 'exp':''}
}[platform.system()]

CLANG_PROGRAM_HEADER = '#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half __fp16\n#define uchar unsigned char\n#define bool uchar\n'
class ClangProgram:
  def __init__(self, name:str, prg:str):
    prg = CLANG_PROGRAM_HEADER + prg
    # TODO: is there a way to not write this to disk?
    fn = f"{tempfile.gettempdir()}/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{args['ext']}"
    if not os.path.exists(fn):
      subprocess.check_output(args=('clang -shared -O2 -Wall -Werror -x c '+args['cflags']+' - -o '+fn+'.tmp').split(), input=prg.encode('utf-8'))
      os.rename(fn+'.tmp', fn)
    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name]

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait: st = time.monotonic()
    self.fxn(*[x._buf for x in args])
    if wait: return time.monotonic()-st

renderer = functools.partial(uops_to_cstyle, CStyleLanguage(kernel_prefix=args['exp'], buffer_suffix=" restrict"))
ClangBuffer = Compiled(RawMallocBuffer, LinearizerOptions(supports_float4=False, has_local=False), renderer, ClangProgram)
