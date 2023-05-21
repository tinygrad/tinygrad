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
code = open(pathlib.Path(__file__).parent / "prog.s", "r").read()

gen = []
FLOPS = 0
#MAX_REG = 251
MAX_REG = 32
for j in range(1):
  for i in range(0, MAX_REG, 6):
    #gen.append(f"v_dual_fmac_f32 v{i+0}, v{i+1}, v{i+2} :: v_dual_fmac_f32 v{i+3}, v{i+4}, v{i+5}")
    #FLOPS += 4
    gen.append(f"v_dual_dot2acc_f32_f16 v{i+0}, v{i+1}, v{i+2} :: v_dual_dot2acc_f32_f16 v{i+3}, v{i+4}, v{i+5}")
    FLOPS += 8
code = code.replace("// FLOPS", '\n'.join(gen))
print(code)


# fix: COMGR failed to get code object ISA name. set triple to 'amdgcn-amd-amdhsa'

object = early_exec(([ROCM_LLVM_PATH / "llvm-mc", '--arch=amdgcn', '--mcpu=gfx1100', '--triple=amdgcn-amd-amdhsa', '--filetype=obj', '-'], code.encode("utf-8")))
asm = early_exec(([ROCM_LLVM_PATH / "ld.lld", "/dev/stdin", "-o", "/dev/stdout", "--pie"], object))

with open("/tmp/cc2.o", "wb") as f:
  f.write(object)
with open("/tmp/cc2.elf", "wb") as f:
  f.write(asm)

print(colored("creating CLProgram", "green"))
prg = CLProgram("code", asm, binary=True)

print(colored("running program", "green"))
G = 256
FLOPS *= 100000*G*G  # loop * global_size
for i in range(3):
  tm = prg([G, G], [256, 1], buf, wait=True)
  print(f"ran in {tm*1e3:.2f} ms, {FLOPS/(tm*1e9):.2f} GFLOPS")

print(colored("transferring buffer", "green"))
print(buf.toCPU())
