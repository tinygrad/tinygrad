import os, time, ctypes, hashlib, subprocess, platform, tempfile
from functools import partial, reduce
from tinygrad.ops import Compiled
from tinygrad.helpers import fromimport, getenv, DEBUG, CI
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage
from unicorn import *
import struct
from unicorn.arm64_const import *
import math
import numpy as np
args = {
  'Windows': {'cflags':'', 'ext':'dll', 'exp':'__declspec(dllexport)'},
  'Linux': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'so', 'exp':''},
  'Darwin': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'dylib', 'exp':''}
}[platform.system()]
CLANG_PROGRAM_HEADER = '#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half __fp16\n#define uchar unsigned char\n#define bool uchar\n'
ADDRESS = 0x10000
STACK_ADDR = 0x40000000
SIZE = 20 * 1024 * 1024
mock_lm = {"sinf": np.sin, "sqrtf": np.sqrt, "exp2f": np.exp2, "log2f": np.log2}
# callback for tracing instructions
def hook_code(fn, uc, address, size, user_data):
  s0_float = struct.unpack('f', struct.pack('I', uc.reg_read(UC_ARM64_REG_S0)))[0]
  #print(s0_float, fn, mock_lm[fn](s0_float))
  res = mock_lm[fn](s0_float)
  uc.reg_write(UC_ARM64_REG_S0, struct.unpack('I', struct.pack('f', res))[0])

class ClangProgram:
  def __init__(self, name:str, prg:str, binary:bool=False, var_size:int=0):
    # TODO: is there a way to not write this to disk?
    fn = f"{tempfile.gettempdir()}/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{args['ext']}"
    if not binary:
      prg = CLANG_PROGRAM_HEADER + prg
      # TODO: is there a way to not write this to disk?
      if not os.path.exists(fn):
        subprocess.check_output(args=('clang -shared -O2 -Wall -Werror -x c '+args['cflags']+' - -o '+fn+'.tmp').split(), input=prg.encode('utf-8'))
        os.rename(fn+'.tmp', fn)
    else:
      if DEBUG >= 5: print(prg)
      if CI:
        # Remove headers and ret
        prg = prg.split('\n')[6:-3]
        self.lm_calls = {(i*4+ADDRESS): ins.split(" ")[1] for i, ins in enumerate(prg) if ins[:2] == 'bl'}
        #each instruction 4 bytes
        prg = "\n".join(ins if i*4+ADDRESS not in self.lm_calls else "add xzr,xzr,xzr" for i, ins in enumerate(prg))
        subprocess.check_output(args=('as -o '+fn+'.o').split(), input=prg.encode('utf-8'))
        subprocess.check_output(args=('objcopy -O binary --only-section=.text '+fn+ '.o ' + fn +'.bin').split())
        with open(fn+'.bin', 'rb') as f:
          data = f.read()
        self.prg = data
        return
      subprocess.check_output(args=('as -o '+fn+'.o').split(), input=prg.encode('utf-8'))
      subprocess.check_output(args=('clang -lm -shared -fPIC '+fn+'.o -o'+fn).split())
    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name]

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait: st = time.monotonic()
    if not CI:
      self.fxn(*[x._buf for x in args])
    else:
      try: 
        mu = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
        reserve_mem = reduce(lambda total, arg: total + arg.size * arg.dtype.itemsize, args, 0)
        #reserve_stack = self.stack_size 
        mu.mem_map(ADDRESS, (reserve_mem + len(self.prg) + 4095) & ~(4095))
        mu.mem_map(STACK_ADDR, SIZE)
        # write machine code to be emulated to memory
        mu.mem_write(ADDRESS, self.prg)
        for k,v in self.lm_calls.items():
          mu.hook_add(UC_HOOK_CODE, partial(hook_code, v), begin=k, end=k)

        addr = ADDRESS + len(self.prg)
        to_stack = []
        for i in range(len(args)):
          mu.mem_write(addr, args[i]._buffer().tobytes())
          if i<=7: 
            mu.reg_write(getattr(unicorn.arm64_const, f'UC_ARM64_REG_X{i}'), addr)
          else:
            to_stack.append(addr.to_bytes(8, 'little'))
          addr += args[i].size * args[i].dtype.itemsize 

        for i, addr in enumerate(to_stack):
          mu.mem_write((STACK_ADDR+SIZE-(len(args[8:])+1)*8) + ((8*i)), addr)

        mu.reg_write(UC_ARM64_REG_SP, STACK_ADDR+SIZE-(len(args[8:])+1)*8)
        mu.emu_start(ADDRESS, ADDRESS + len(self.prg))
        args[0]._buf = mu.mem_read(mu.reg_read(UC_ARM64_REG_X0), args[0].size * args[0].dtype.itemsize)
      except UcError as e:
        print("ERROR: %s" % e)
    if wait: return time.monotonic()-st

class ClangCodegen(CStyleCodegen):
  lang = CStyleLanguage(kernel_prefix=args['exp'], buffer_suffix=" restrict")
  supports_float4: bool = False

ClangBuffer = Compiled(RawMallocBuffer, fromimport("extra.assembly.assembly_arm64", "ARM64Codegen") if getenv("ARM64") else ClangCodegen, ClangProgram)