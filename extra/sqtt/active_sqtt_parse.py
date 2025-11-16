import atexit
from tinygrad import Device
from tinygrad.helpers import system
from extra.sqtt.test_timing import save_sqtt
from tinygrad.runtime.ops_amd import AMDProgram

def set_power(x): system(f"sudo /opt/rocm/bin/amd-smi set -l {x}")
@atexit.register
def reset_power(): set_power("auto")

DEV = Device["AMD"]

template = """.text
.globl matmul
.p2align 8
.type matmul,@function
matmul:
  INSTRUCTION
  s_endpgm

.rodata
.p2align 6
.amdhsa_kernel matmul
  .amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: matmul
    .symbol: matmul.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 32
    .max_flat_workgroup_size: 1024
...
.end_amdgpu_metadata
"""

def run_asm(src):
  NUM_WORKGROUPS = 1
  WAVE_SIZE = 32
  NUM_WAVES = 1
  lib = DEV.compiler.compile(template.replace("INSTRUCTION", src))
  fxn = AMDProgram(DEV, "matmul", lib)
  fxn(global_size=(NUM_WORKGROUPS,1,1), local_size=(WAVE_SIZE*NUM_WAVES,1,1), wait=True)

if __name__ == "__main__":
  set_power("stable_std")
  with save_sqtt() as sqtt:
    run_asm("s_nop 1")
