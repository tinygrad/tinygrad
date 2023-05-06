import pathlib
from hexdump import hexdump
from extra.helpers import enable_early_exec
early_exec = enable_early_exec()

"""
prg_empty = CLProgram("code", "__kernel void code() { }")
asm_real = prg_empty.binary()
with open("/tmp/cc.elf", "wb") as f:
  f.write(asm_real)
"""

code = b"""
.text
code:
s_endpgm
.amdgpu_metadata
amdhsa.kernels:
  - .args:           []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .kernarg_segment_size: 0
    .language:       OpenCL C
    .language_version:
      - 1
      - 2
    .max_flat_workgroup_size: 256
    .name:           code
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         code.kd
    .uses_dynamic_stack: false
    .vgpr_count:     0
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1100
amdhsa.version:
  - 1
  - 2
.end_amdgpu_metadata
"""

# fix: COMGR failed to get code object ISA name. set triple to 'amdgcn-amd-amdhsa'
object = early_exec(([pathlib.Path(__file__).parent.parent.parent.parent / "extra/rocm/build/llvm-project/bin/llvm-mc", '--arch=amdgcn', '--mcpu=gfx1100', '--triple=amdgcn-amd-amdhsa', '--filetype=obj', '-'], code))
asm = early_exec(([pathlib.Path(__file__).parent.parent.parent.parent / "extra/rocm/build/llvm-project/bin/ld.lld", "/dev/stdin", "-o", "/dev/stdout", "--pie"], object))

with open("/tmp/cc2.o", "wb") as f:
  f.write(object)
with open("/tmp/cc2.elf", "wb") as f:
  f.write(asm)

from tinygrad.runtime.ops_gpu import CLProgram
prg = CLProgram("code", asm, binary=True)

