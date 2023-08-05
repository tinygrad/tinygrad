import os, time, ctypes, hashlib, subprocess, platform, tempfile
from functools import partial
from tinygrad.ops import Compiled
from tinygrad.helpers import fromimport, getenv, DEBUG, CI
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage
from unicorn import *
import struct
from unicorn.arm64_const import *
from keystone import *
import math
import numpy as np
args = {
  'Windows': {'cflags':'', 'ext':'dll', 'exp':'__declspec(dllexport)'},
  'Linux': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'so', 'exp':''},
  'Darwin': {'cflags':'-lm -fPIC --rtlib=compiler-rt ', 'ext':'dylib', 'exp':''}
}[platform.system()]
CLANG_PROGRAM_HEADER = '#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half __fp16\n#define uchar unsigned char\n#define bool uchar\n'
ADDRESS = 0x10000
mock_lm = {"sinf": np.sin, "sqrtf": np.sqrt, "exp2f": np.exp2, "log2f": np.log2}

# callback for tracing instructions
def hook_code(external_calls, uc, address, size, user_data):
  if address in external_calls:
    s0 = uc.reg_read(UC_ARM64_REG_S0)
    float_number = mock_lm[external_calls[address]](struct.unpack('!f', bytes.fromhex(hex(s0)[2:]))[0])
    print(mock_lm[external_calls[address]])
    uc.reg_write(UC_ARM64_REG_S0, struct.unpack('I', struct.pack('f', float_number))[0])

class ClangProgram:
  def __init__(self, name:str, prg:str, binary:bool=False):
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
        prg = prg.split('\n')[5:-2]
        matches = {(i*4+ADDRESS): s.split(" ")[1] for i,s in enumerate(prg) if s.find('bl') != -1}
        #each instruction 4 bytes
        prg = "\n".join(line if line.find('bl') == -1 else "mov x10, xzr" for line in prg)
        # Convert to bytes
        ks = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
        output, count = ks.asm(prg)
        #subprocess.check_output(args=('as -o '+fn+'.o').split(), input=prg.encode('utf-8'))
        #subprocess.check_output(args=('clang -lm -shared -fPIC '+fn+'.o -o'+fn).split())
        output = bytes(output)
        mu = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
        # map 2MB memory for this emulation
        mu.mem_map(ADDRESS, 100 * 1024 * 1024)
        # write machine code to be emulated to memory
        mu.mem_write(ADDRESS, output)
        STACK_ADDR = 0x10000000
        STACK_SIZE = 1024 * 1024
        mu.mem_map(STACK_ADDR, STACK_SIZE)
        mu.reg_write(UC_ARM64_REG_SP, STACK_ADDR + STACK_SIZE - 0x10)
        self.start = ADDRESS 
        self.end = ADDRESS + len(output) 
        self.mu = mu
    # self.lib = ctypes.CDLL(fn)
    # self.fxn = self.lib[name]

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait: st = time.monotonic()
    if not CI:
      self.fxn(*[x._buf for x in args])
    else:
      try:
        regs = [getattr(unicorn.arm64_const, f'UC_ARM64_REG_X{i}') for i in range(len(args))]
        addr = self.end + 4 
        for i, x in enumerate(args):
          self.mu.mem_write(addr, args[i]._buffer().tobytes())
          self.mu.reg_write(regs[i], addr)
          addr += args[i].size * args[i].dtype.itemsize 

        self.mu.emu_start(self.start, self.end)
        x0 = self.mu.reg_read(UC_ARM64_REG_X0)
        val = self.mu.mem_read(x0, args[0].size * args[0].dtype.itemsize)
        args[0]._buf = val 
      except UcError as e:
        print("ERROR: %s" % e)
    if wait: return time.monotonic()-st

class ClangCodegen(CStyleCodegen):
  lang = CStyleLanguage(kernel_prefix=args['exp'], buffer_suffix=" restrict")
  supports_float4: bool = False

ClangBuffer = Compiled(RawMallocBuffer, fromimport("extra.assembly.assembly_arm64", "ARM64Codegen") if getenv("ARM64") else ClangCodegen, ClangProgram)
