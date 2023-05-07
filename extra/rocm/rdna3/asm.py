import numpy as np
import pathlib
from hexdump import hexdump
from tinygrad.helpers import colored
from extra.helpers import enable_early_exec
early_exec = enable_early_exec()

from tinygrad.runtime.ops_gpu import CLProgram, CLBuffer, ROCM_LLVM_PATH

ENABLE_NON_ASM = False

if ENABLE_NON_ASM:
  buf = CLBuffer.fromCPU(np.zeros(10, np.float32))
  prg_empty = CLProgram("code", "__kernel void code(__global float *a) { a[0] = 1; }")
  asm_real = prg_empty.binary()
  with open("/tmp/cc.elf", "wb") as f:
    f.write(asm_real)
  prg_empty([1], [1], buf, wait=True)
  print(buf.toCPU())

print(colored("creating CLBuffer", "green"))
buf = CLBuffer.fromCPU(np.zeros(10, np.float32))
code = open(pathlib.Path(__file__).parent / "prog.s", "rb").read()

# fix: COMGR failed to get code object ISA name. set triple to 'amdgcn-amd-amdhsa'

object = early_exec(([ROCM_LLVM_PATH / "llvm-mc", '--arch=amdgcn', '--mcpu=gfx1100', '--triple=amdgcn-amd-amdhsa', '--filetype=obj', '-'], code))
asm = early_exec(([ROCM_LLVM_PATH / "ld.lld", "/dev/stdin", "-o", "/dev/stdout", "--pie"], object))

with open("/tmp/cc2.o", "wb") as f:
  f.write(object)
with open("/tmp/cc2.elf", "wb") as f:
  f.write(asm)

print(colored("creating CLProgram", "green"))
prg = CLProgram("code", asm, binary=True)

print(colored("running program", "green"))
prg([1], [1], buf, wait=True)

print(colored("transferring buffer", "green"))
print(buf.toCPU())
