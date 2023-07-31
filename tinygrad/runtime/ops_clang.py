import os, time, ctypes, hashlib, subprocess, platform, tempfile, portalocker
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

args = {
  'Windows': {'cflags':'', 'ext':'dll', 'exp':'__declspec(dllexport)'},
  'Linux': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'so', 'exp':''},
  'Darwin': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'dylib', 'exp':''}
}[platform.system()]

def _compile_to_lib(prg:str):
  # TODO: is there a way to not write this to disk?
  dst = f"{tempfile.gettempdir()}/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{args['ext']}"
  with portalocker.Lock(dst+'.lock', flags=portalocker.LockFlags.EXCLUSIVE) as lock:
    if not os.path.exists(dst):
      subprocess.check_output(args=('clang -shared -O2 -Wall -Werror -march=native -x c '+args['cflags']+' - -o '+dst+'.tmp').split(), input=prg.encode('utf-8'))
      os.rename(dst+'.tmp', dst)
    return ctypes.CDLL(dst)

if platform.system() == 'Linux':
  # Reproducer taken from https://github.com/llvm/llvm-project/issues/56204
  lib = _compile_to_lib('float __attribute__((noinline)) n(float v) { return -v; } int check() { return (__fp16)n(-1.f); }')
  if lib['check']() != 1:
    raise RuntimeError('Test lib incorrectly handles half precision floating point. CLANG backend won\'t produce valid results')

class ClangProgram:
  def __init__(self, name:str, prg:str):
    prg = '#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half __fp16\n#define uchar unsigned char\n#define bool uchar\n' + prg
    self.lib = _compile_to_lib(prg)
    self.fxn = self.lib[name]

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait: st = time.monotonic()
    self.fxn(*[x._buf for x in args])
    if wait: return time.monotonic()-st

class ClangCodegen(CStyleCodegen):
  lang = CStyleLanguage(kernel_prefix=args['exp'], buffer_suffix=" restrict")
  supports_float4: bool = False

ClangBuffer = Compiled(RawMallocBuffer, ClangCodegen, ClangProgram)
