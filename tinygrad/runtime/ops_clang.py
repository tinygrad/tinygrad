import time, ctypes, hashlib, subprocess, platform, functools
from functools import partial, reduce, lru_cache
from tinygrad.ops import Compiled
from tinygrad.helpers import fromimport, getenv, DEBUG, CI
from tinygrad.runtime.lib import CachedProgram, RawMallocBuffer
from tinygrad.codegen.linearizer import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage
import struct
import numpy as np

ARM64 = getenv('ARM64', False)
if CI and ARM64: from unicorn import Uc, UC_ARCH_ARM64, UC_MODE_ARM, UC_HOOK_CODE, arm64_const   # type: ignore

args = {
  'Windows': {'cflags':'', 'ext':'dll', 'exp':'__declspec(dllexport)'},
  'Linux': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'so', 'exp':''},
  'Darwin': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'dylib', 'exp':''}
}[platform.system()]

CLANG_PROGRAM_HEADER = '#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half __fp16\n#define uchar unsigned char\n#define bool uchar\n'
ADDRESS = 0x10000

# Unicorn doesn't support external calls
def align(addr): return (addr+4095) & ~(4095)
mock_lm = {"sinf": np.sin, "sqrtf": np.sqrt, "exp2f": np.exp2, "log2f": np.log2}
def emulate_ext_calls(fn, uc, address, size, user_data):
  s_in = struct.unpack('f', struct.pack('I', uc.reg_read(getattr(arm64_const, f'UC_ARM64_REG_S{fn[2][1:]}'))))[0]
  uc.reg_write(getattr(arm64_const, f'UC_ARM64_REG_S{fn[1][1:]}'), struct.unpack('I', struct.pack('f', mock_lm[fn[0]](s_in)))[0])  # type: ignore

class ClangProgram(CachedProgram):
  def __init__(self, name:str, prg:str, binary:bool=False):
    if binary and DEBUG >= 5: print(prg)
    extension = args['ext'] if not binary else 'bin'
    CachedProgram.__init__(self, name, prg, binary=binary, extension=extension)
    if binary and CI and ARM64:
      prg_lines = prg.split('\n')
      self.varsize = align(int(prg_lines[0].split(" ")[1]))
      self.ext_calls = {(i*4+ADDRESS):ins.split(" ")[1:] for i, ins in enumerate(filter(lambda ins: ins[:4] != 'loop', prg_lines[6:-3])) if ins[:2] == 'bl'}
      with open(self.bin_cache_path, "rb") as f:
        self.prg = f.read()
    else:
      self.lib = ctypes.CDLL(self.bin_cache_path)
      self.fxn = self.lib[name]
  def compile(self, name:str, prg:str, binary:bool=False):
    # TODO: is there a way to not write this to disk?
    # A: it seems there isn't https://stackoverflow.com/questions/28053328/ctypes-cdll-load-library-from-memory-rather-than-file
    #    because ctypes.CDLL() calls dlopen (POSIX) or LoadLibrary (Windows) which require a file
    if not binary:
      prg = CLANG_PROGRAM_HEADER + prg
      subprocess.check_output(args=('clang -shared -O2 -Wall -Werror -x c '+args['cflags']+' - -o '+self.bin_cache_path_tmp).split(), input=prg.encode('utf-8'))
    else:
      if CI and ARM64:
        prg_lines = prg.split('\n')
        prg = "\n".join(['nop' if ins[:2] == 'bl' else ins for ins in prg_lines[6:-3]] + ['\n'])
        subprocess.check_output(args=(f'aarch64-linux-gnu-as -o {self.bin_cache_path_tmp_as}').split(), input=prg.encode('utf-8'))
        subprocess.check_output(args=(f'aarch64-linux-gnu-objcopy -O binary --only-section=.text {self.bin_cache_path_tmp_as} {self.bin_cache_path_tmp}').split())
      else:
        subprocess.check_output(args=(f'as -o {self.bin_cache_path_tmp_as}').split(), input=prg.encode('utf-8'))
        subprocess.check_output(args=(f'clang -lm -shared {self.bin_cache_path_tmp_as} {self.bin_cache_path}').split())
  @staticmethod
  @lru_cache
  def toolchain_hash():
    version_cmd = "clang --version"
    if ARM64:
      version_cmd = "aarch64-linux-gnu-as --version"
    version_hash = hashlib.sha256(subprocess.check_output(args=(version_cmd).split())).digest().hex()
    return f"clang-{version_hash}"
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

renderer = fromimport("tinygrad.codegen.assembly_arm64", "uops_to_arm64_asm") if ARM64 else functools.partial(uops_to_cstyle, CStyleLanguage(kernel_prefix=args['exp'], buffer_suffix=" restrict", arg_int_prefix="const int"))
ClangBuffer = Compiled(RawMallocBuffer, LinearizerOptions(supports_float4=False, has_local=False), renderer, ClangProgram)
