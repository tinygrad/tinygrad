from tinygrad.codegen.linearizer import Linearizer
from tinygrad.ops import ASTRunner
from tinygrad.runtime.ops_gpu import ROCM_LLVM_PATH

# ugh, is this really needed?
from extra.helpers import enable_early_exec
early_exec = enable_early_exec()

# https://github.com/ROCm-Developer-Tools/ROCm-ComputeABI-Doc/blob/master/AMDGPU-ABI.md#initial-kernel-register-state
# enable_sgpr_kernarg_segment_ptr
# enable_sgpr_grid_workgroup_count_X

# amd_kernel_..., amd_machine_...
# kernel_code_entry_byte_offset, kernel_code_prefetch_byte_offset
# kernel_code_prefetch_byte_size, max_scratch_backing_memory_byte_size
# compute_pgm_rsrc1, compute_pgm_rsrc2, kernel_code_properties, workitem_private_segment_byte_size

# TODO: generate this struct
boilerplate_start = """
.global _start
_start:
.rodata
.align 0x10
.global code.kd
.type code.kd,STT_OBJECT
code.kd:
.long 0,0,0,0
.long 0x00000bc0,0x00000000,0x00000000,0x00000000
.long 0,0,0,0
.long 0x60af0000,0x0000009e,0x00000408,0x00000000
.text
"""

# TODO: generate this yaml
boilerplate_end = """
.amdgpu_metadata
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .name:           a
        .offset:         0
        .size:           8
        .type_name:      'float*'
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           b
        .offset:         0
        .size:           8
        .type_name:      'float*'
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           c
        .offset:         0
        .size:           8
        .type_name:      'float*'
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .language:       OpenCL C
    .language_version:
      - 1
      - 2
    .max_flat_workgroup_size: 256
    .name:           code
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .sgpr_spill_count: 0
    .symbol:         code.kd
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1100
amdhsa.version:
  - 1
  - 2
.end_amdgpu_metadata
"""

class AssemblyCodegen(Linearizer):
  supports_float4: bool = True

  # s registers are the addresses and non local indexes
  def codegen(self):
    self.process()
    self.hand_coded_optimizations()
    self.linearize()

    instructions = []

    # exit asm
    instructions += ['s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)', 's_endpgm', 's_code_end']

    code = boilerplate_start + '\n'.join(instructions) + boilerplate_end
    object = early_exec(([ROCM_LLVM_PATH / "llvm-mc", '--arch=amdgcn', '--mcpu=gfx1100', '--triple=amdgcn-amd-amdhsa', '--filetype=obj', '-'], code.encode("utf-8")))
    asm = early_exec(([ROCM_LLVM_PATH / "ld.lld", "/dev/stdin", "-o", "/dev/stdout", "--pie"], object))

    global_size = []
    local_size = []
    return ASTRunner('code', asm,
      global_size[::-1] if len(global_size) else [1], local_size[::-1] if len(local_size) else None,
      op_estimate=self.info.flops, mem_estimate=self.mem_estimate, display_name=self.function_name, runtime_args={"binary": True})
