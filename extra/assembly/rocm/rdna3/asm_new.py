import pathlib, subprocess
from hexdump import hexdump
from tinygrad.runtime.driver.hip_comgr import compile_hip
import yaml

boilerplate_start = """
.global _start
_start:
.rodata
.align 0x10
.global code.kd
.type code.kd,STT_OBJECT
.amdhsa_kernel code"""

code_start = """.end_amdhsa_kernel
.text
code:
"""

args = []
i = 0
args.append({'.address_space': 'global', '.name': f'buf_{i}', '.offset': i*8, '.size': 8, '.type_name': "float*", '.value_kind': 'global_buffer'})
v_cnt = 0
s_cnt = 0

kernel_desc = {'.amdhsa_group_segment_fixed_size': 0, '.amdhsa_private_segment_fixed_size': 0, '.amdhsa_kernarg_size': 0,
            '.amdhsa_next_free_vgpr': v_cnt,   # this matters!
            '.amdhsa_reserve_vcc': 0, '.amdhsa_reserve_xnack_mask': 0,
            '.amdhsa_next_free_sgpr': s_cnt,
            '.amdhsa_float_round_mode_32': 0, '.amdhsa_float_round_mode_16_64': 0, '.amdhsa_float_denorm_mode_32': 3, '.amdhsa_float_denorm_mode_16_64': 3, '.amdhsa_dx10_clamp': 1, '.amdhsa_ieee_mode': 1,
            '.amdhsa_fp16_overflow': 0, '.amdhsa_workgroup_processor_mode': 1, '.amdhsa_memory_ordered': 1, '.amdhsa_forward_progress': 0, '.amdhsa_enable_private_segment': 0,
            '.amdhsa_system_sgpr_workgroup_id_x': 1, '.amdhsa_system_sgpr_workgroup_id_y': 1, '.amdhsa_system_sgpr_workgroup_id_z': 1,
            '.amdhsa_system_sgpr_workgroup_info': 0, '.amdhsa_system_vgpr_workitem_id': 2, # is amdhsa_system_vgpr_workitem_id real?
            '.amdhsa_exception_fp_ieee_invalid_op': 0, '.amdhsa_exception_fp_denorm_src': 0, '.amdhsa_exception_fp_ieee_div_zero': 0, '.amdhsa_exception_fp_ieee_overflow': 0, '.amdhsa_exception_fp_ieee_underflow': 0,
            '.amdhsa_exception_fp_ieee_inexact': 0, '.amdhsa_exception_int_div_zero': 0, '.amdhsa_user_sgpr_dispatch_ptr': 0, '.amdhsa_user_sgpr_queue_ptr': 0, '.amdhsa_user_sgpr_kernarg_segment_ptr': 1,
            '.amdhsa_user_sgpr_dispatch_id': 0, '.amdhsa_user_sgpr_private_segment_size': 0, '.amdhsa_wavefront_size32': 1, '.amdhsa_uses_dynamic_stack': 0}

metadata = {'amdhsa.kernels': [{'.args': args,
          '.group_segment_fixed_size': 0, '.kernarg_segment_align': 8, '.kernarg_segment_size': args[-1][".offset"] + args[-1][".size"],
          '.language': 'OpenCL C', '.language_version': [1, 2], '.max_flat_workgroup_size': 256,
          '.name': 'code', '.private_segment_fixed_size': 0, '.sgpr_count': s_cnt, '.sgpr_spill_count': 0,
          '.symbol': 'code.kd', '.uses_dynamic_stack': False, '.vgpr_count': v_cnt, '.vgpr_spill_count': 0,
          '.wavefront_size': 32}],
        'amdhsa.target': 'amdgcn-amd-amdhsa--gfx1100', 'amdhsa.version': [1, 2]}

ROCM_LLVM_PATH = pathlib.Path("/opt/rocm/lib/llvm/bin")

ins = ['s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)', 's_endpgm', 's_code_end']

if __name__ == "__main__":
  code = boilerplate_start + "\n" + '\n'.join("%s %d" % x for x in kernel_desc.items()) + "\n" +  code_start + '\n'.join(ins) + "\n.amdgpu_metadata\n" + yaml.dump(metadata) + ".end_amdgpu_metadata"
  out = compile_hip(code, asm=True)
  hexdump(out[:0x80])
  #obj = subprocess.check_output([ROCM_LLVM_PATH / "llvm-mc", '--arch=amdgcn', '--mcpu=gfx1100', '--triple=amdgcn-amd-amdhsa', '--filetype=obj', '-'], input=code.encode("utf-8"))
  #hexdump(obj)

