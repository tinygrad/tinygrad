import os, time, ctypes, hashlib, subprocess, platform
from tinygrad.helpers import dtypes, DType
from tinygrad.ops import CompiledBuffer, RawBufferMapped, Specialized
from tinygrad.codegen.gpu import GPUCodegen, GPULanguage

class RawMallocBuffer(RawBufferMapped):
  def __init__(self, size, dtype: DType):
    super().__init__(size, dtype)
    self._buf = ({dtypes.float32: ctypes.c_float, dtypes.float16: ctypes.c_int16}[dtype] * size)()
  def _buffer(self): return memoryview(self._buf)

class ClangProgram:
  def __init__(self, name:str, prg:str):
    prg = "#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define half __fp16\n" + prg
    # TODO: is there a way to not write this to disk?
    fn = f"/tmp/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{'dylib' if platform.system() == 'Darwin' else 'so'}"
    # NOTE: --rtlib=compiler-rt fixes float16 on Linux, it defines __gnu_h2f_ieee and __gnu_f2h_ieee
    if not os.path.exists(fn):
      subprocess.check_output(['clang', '-shared', '-O2', '-Wall','-Werror', '-lm', '--rtlib=compiler-rt', '-fPIC', '-x', 'c', '-', '-o', fn+".tmp"], input=prg.encode('utf-8'))
      os.rename(fn+".tmp", fn)
    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name]
  def __call__(self, *args, wait=False):
    if wait: st = time.monotonic()
    self.fxn(*[x._buf for x in args[2:]])
    if wait: return time.monotonic()-st

class ClangCodegen(GPUCodegen):
  lang = GPULanguage(buffer_suffix=" restrict")

class ClangBuffer(CompiledBuffer):
  spec = Specialized(RawMallocBuffer, ClangCodegen, ClangProgram)
