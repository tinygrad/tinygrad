import os, time, ctypes, hashlib, subprocess, platform, tempfile
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

cfg = {
  # NOTE: --rtlib=compiler-rt fixes float16 on Linux, it defines __gnu_h2f_ieee and __gnu_f2h_ieee
  'base': {'cflags':'clang -shared -O2 -Wall -Werror --rtlib=compiler-rt -x c', 'ext':'', 'export':''},
  'Windows': {'cflags':'', 'ext':'dll', 'export':'__declspec(dllexport)'},
  'Linux': {'cflags':'-lm -fPIC', 'ext':'so', 'export':'__attribute__((visibility("default")))'},
  'Darwin': {'cflags':'', 'ext':'dylib', 'export':''}
}
plat_cfg = cfg[platform.system()]

class ClangProgram:
  def __init__(self, name:str, prg:str):
    prg = "#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half __fp16\n#define uchar unsigned char\n#define bool uchar\n" + prg
    # TODO: is there a way to not write this to disk?
    fn = f"{tempfile.gettempdir()}/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{plat_cfg['ext']}"
    if not os.path.exists(fn):
      subprocess.check_output( (cfg['base']['cflags']+' '+plat_cfg['cflags']+ ' -o ' + fn+ '.tmp -').split(), input=prg.encode('utf-8'))
      os.rename(fn+".tmp", fn)
    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name]

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait: st = time.monotonic()
    self.fxn(*[x._buf for x in args])
    if wait: return time.monotonic()-st

class ClangCodegen(CStyleCodegen):
  lang = CStyleLanguage(kernel_prefix=plat_cfg['export'], buffer_suffix=" restrict")
  supports_float4: bool = False

ClangBuffer = Compiled(RawMallocBuffer, ClangCodegen, ClangProgram)
