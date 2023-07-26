import os, time, ctypes, hashlib, subprocess, platform, tempfile
from tinygrad.ops import Compiled
from tinygrad.helpers import fromimport, getenv, DEBUG
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

args = {
  'Windows': {'cflags':'', 'ext':'dll', 'exp':'__declspec(dllexport)'},
  'Linux': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'so', 'exp':''},
  'Darwin': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'dylib', 'exp':''}
}[platform.system()]

class ClangProgram:
  def __init__(self, name:str, prg:str, binary:bool=False):
    fn = f"{tempfile.gettempdir()}/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{args['ext']}"
    if not binary:
      prg = '#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half __fp16\n#define uchar unsigned char\n#define bool uchar\n' + prg
      # TODO: is there a way to not write this to disk?
      if not os.path.exists(fn):
        subprocess.check_output(args=('clang -shared -O2 -Wall -Werror -x c '+args['cflags']+' - -o '+fn+'.tmp').split(), input=prg.encode('utf-8'))
        os.rename(fn+'.tmp', fn)
    else:
      if DEBUG >= 5: print(prg)
      """
     .arch armv8-a
.text
.global _test
.p2align 2
_test:
Lloh0:
        adrp    x8, _sinf@GOTPAGE
        ldr     s0, [x1]
        mov     x19, x0
Lloh1:
        ldr     x8, [x8, _sinf@GOTPAGEOFF]
        blr     x8
        ldp     x29, x30, [sp, #16]             ; 16-byte Folded Reload
        str     s0, [x19]
        ret
      """
      if getenv('ARM64'):
        subprocess.check_output(args=('as -arch arm64 -o '+fn+'.o').split(), input=prg.encode('utf-8'))
        subprocess.check_output(args=('clang -lm -O2 -Wall -shared '+fn+'.o -o'+fn).split())
    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name]

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait: st = time.monotonic()
    self.fxn(*[x._buf for x in args])
    if wait: return time.monotonic()-st

class ClangCodegen(CStyleCodegen):
  lang = CStyleLanguage(kernel_prefix=args['exp'], buffer_suffix=" restrict")
  supports_float4: bool = False

ClangBuffer = Compiled(RawMallocBuffer, fromimport("extra.assembly.assembly_arm64", "ARM64Codegen") if getenv("ARM64") else ClangCodegen, ClangProgram)
