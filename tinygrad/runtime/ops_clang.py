import time, ctypes, subprocess, platform, functools, pathlib, tempfile
from typing import Any
from tinygrad.ops import Compiled
from tinygrad.helpers import cache_compiled
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage

args = {
  'Linux': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'so'},
  'Darwin': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'dylib'}
}[platform.system()]

CLANG_PROGRAM_HEADER = '#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half __fp16\n#define uchar unsigned char\n#include <stdbool.h>\n'

class ClangProgram:
  def __init__(self, name:str, prg:str, binary=False):
    self.prg: bytes = self.compile(CLANG_PROGRAM_HEADER+prg)

    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(self.prg)
      self.fxn: Any = ctypes.CDLL(str(cached_file_path.name))[name]

  @cache_compiled
  def compile(self, prg) -> bytes:
    return subprocess.check_output(args=('clang -shared -O2 -Wall -Werror -x c '+args['cflags']+' - -o /dev/stdout').split(), input=prg.encode('utf-8'))

  def __call__(self, unused_global_size, unused_local_size, *args, wait=False):
    if wait: st = time.monotonic()
    self.fxn(*[x._buf if isinstance(x, RawMallocBuffer) else x for x in args])
    if wait: return time.monotonic()-st

renderer = functools.partial(uops_to_cstyle, CStyleLanguage(buffer_suffix=" restrict", arg_int_prefix="const int"))
ClangBuffer = Compiled(RawMallocBuffer, LinearizerOptions(supports_float4=False, has_local=False), renderer, ClangProgram)
