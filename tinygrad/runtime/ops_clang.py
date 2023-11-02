import time, ctypes, subprocess, platform, functools, pathlib, tempfile
from typing import Any
from functools import partial, reduce
from tinygrad.ops import Compiled
from tinygrad.helpers import fromimport, getenv, DEBUG, CI, cache_compiled
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage
import struct
import numpy as np

ARM64 = getenv('ARM64', False)
if CI and ARM64: from unicorn import Uc, UC_ARCH_ARM64, UC_MODE_ARM, UC_HOOK_CODE, arm64_const   # type: ignore

args = {
  'Windows': {'cflags':'', 'ext':'dll', 'exp':'__declspec(dllexport) '},
  'Linux': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'so', 'exp':''},
  'Darwin': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'dylib', 'exp':''}
}[platform.system()]

CLANG_PROGRAM_HEADER = '#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half __fp16\n#define uchar unsigned char\n#include <stdbool.h>\n'
ADDRESS = 0x10000

# Unicorn doesn't support external calls
def align(addr): return (addr+4095) & ~(4095)
mock_lm = {"sinf": np.sin, "sqrtf": np.sqrt, "exp2f": np.exp2, "log2f": np.log2}
def emulate_ext_calls(fn, uc, address, size, user_data):
  s_in = struct.unpack('f', struct.pack('I', uc.reg_read(getattr(arm64_const, f'UC_ARM64_REG_S{fn[2][1:]}'))))[0]
  uc.reg_write(getattr(arm64_const, f'UC_ARM64_REG_S{fn[1][1:]}'), struct.unpack('I', struct.pack('f', mock_lm[fn[0]](s_in)))[0])  # type: ignore

class ClangProgram:
  def __init__(self, name:str, prg:str, binary:bool=False):
    if binary and DEBUG >= 5: print(prg)
    self.prg: Any = self.compile(prg if binary else CLANG_PROGRAM_HEADER+prg, binary)

    # TODO: is there a way to not write this to disk?
    # A: it seems there isn't https://stackoverflow.com/questions/28053328/ctypes-cdll-load-library-from-memory-rather-than-file
    #    because ctypes.CDLL() calls dlopen (POSIX) or LoadLibrary (Windows) which require a file
    if not (CI and ARM64):
      with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
        pathlib.Path(cached_file_path.name).write_bytes(self.prg)
        self.fxn: Any = ctypes.CDLL(str(cached_file_path.name))[name]

  @cache_compiled
  def compile(self, prg, binary) -> bytes:
    with tempfile.NamedTemporaryFile(delete=True) as output_file, tempfile.NamedTemporaryFile(delete=True) as temp_file:
      if not binary:
        subprocess.check_output(args=('clang -shared -O2 -Wall -Werror -x c '+args['cflags']+' - -o '+str(output_file.name)).split(), input=prg.encode('utf-8'))
      elif CI and ARM64:
        prg = prg.split('\n') # type: ignore
        self.varsize = align(int(prg[0].split(" ")[1]))
        self.ext_calls = {(i*4+ADDRESS):ins.split(" ")[1:] for i, ins in enumerate(filter(lambda ins: ins[:4] != 'loop', prg[6:-3])) if ins[:2] == 'bl'}
        prg = "\n".join(['nop' if ins[:2] == 'bl' else ins for ins in prg[6:-3]] + ['\n'])
        subprocess.check_output(args=('aarch64-linux-gnu-as -o '+str(temp_file.name)).split(), input=prg.encode('utf-8'))
        subprocess.check_output(args=('aarch64-linux-gnu-objcopy -O binary --only-section=.text '+str(temp_file.name)+' '+str(output_file.name)).split())
      else:
        subprocess.check_output(args=('as -o' + str(temp_file.name)).split(), input=prg.encode('utf-8'))
        subprocess.check_output(args=('clang -lm -shared '+str(temp_file.name)+' -o'+str(output_file.name)).split())
      return pathlib.Path(output_file.name).read_bytes()

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait: st = time.monotonic()
    if CI and ARM64:
      mu = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
      total_mem = align(reduce(lambda total, arg: total + arg.size * arg.dtype.itemsize, args, len(self.prg)+self.varsize))
      mu.mem_map(ADDRESS, total_mem)
      for k, fn in self.ext_calls.items(): mu.hook_add(UC_HOOK_CODE, partial(emulate_ext_calls, fn), begin=k, end=k)
      mu.mem_write(ADDRESS, self.prg + b''.join(bytes(arg._buf) for arg in args))
      addr = ADDRESS + len(self.prg)
      for i, arg in enumerate(args):
        if i<=7:
          mu.reg_write(getattr(arm64_const, f'UC_ARM64_REG_X{i}'), addr)
        else:
          # NOTE: In ARM, args beyond the first 8 are placed on the stack it also account for the stack red zone.
          mu.mem_write(ADDRESS + total_mem - (len(args[8:])+2)*8 + 8*(i-8), addr.to_bytes(8, 'little'))
        addr += arg.size * arg.dtype.itemsize
      mu.reg_write(arm64_const.UC_ARM64_REG_SP, ADDRESS + total_mem - (len(args[8:])+2)*8)
      mu.emu_start(ADDRESS, ADDRESS + len(self.prg))
      args[0]._buf = mu.mem_read(mu.reg_read(arm64_const.UC_ARM64_REG_X0), args[0].size * args[0].dtype.itemsize)
    else:
      self.fxn(*[x._buf if isinstance(x, RawMallocBuffer) else x for x in args])
    if wait: return time.monotonic()-st

renderer = fromimport("tinygrad.renderer.assembly_arm64", "uops_to_arm64_asm") if ARM64 else functools.partial(uops_to_cstyle, CStyleLanguage(kernel_prefix=args['exp'], buffer_suffix=" restrict", arg_int_prefix="const int"))
ClangBuffer = Compiled(RawMallocBuffer, LinearizerOptions(supports_float4=False, has_local=False), renderer, ClangProgram)
