import yaml
from typing import DefaultDict, Tuple
from tinygrad.helpers import dtypes
from tinygrad.codegen.assembly import AssemblyCodegen, Register
from tinygrad.codegen.linearizer import UOps
from tinygrad.ops import BinaryOps, UnaryOps, FusedOps
from tinygrad.runtime.ops_gpu import ROCM_LLVM_PATH
from collections import defaultdict

# ugh, is this really needed?
from extra.helpers import enable_early_exec
early_exec = enable_early_exec()

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

# https://github.com/RadeonOpenCompute/ROCm_Documentation/blob/master/ROCm_Compiler_SDK/ROCm-Codeobj-format.rst
# https://github.com/ROCm-Developer-Tools/ROCm-ComputeABI-Doc/blob/master/AMDGPU-ABI.md#initial-kernel-register-state
# RDNA3 is actually a SIMD machine!
class RDNACodegen(AssemblyCodegen):
  supports_float4: bool = False
  supports_float4_alu: bool = False
  supports_load3: bool = True
  sin_is_sin2pi: bool = True

  def specialize(self, asm) -> Tuple[str, str]:
    args = []
    for i,b in enumerate(self.bufs): args.append({'.address_space': 'global', '.name': f'buf_{i}', '.offset': i*8, '.size': 8, '.type_name': b.dtype.name+"*", '.value_kind': 'global_buffer'})
    ins = []

    v_cnt = 3  # v[0:2] is local_xyz
    s_cnt = 5  # s[0:1] is the address, s[2:4] is global_xyz

    dtype_to_rdnatype = {dtypes.float32: "f32", dtypes.int64: "i64", dtypes.int32: "i32", dtypes.uint64: "u64", dtypes.bool: "i32"}
    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", FusedOps.MULACC: "fma",
           UnaryOps.NOOP: "mov", UnaryOps.SIN: "sin", UnaryOps.LOG2: "log", UnaryOps.EXP2: "exp",
           BinaryOps.CMPEQ: "cmpk_eq", BinaryOps.CMPLT: "cmpk_lt"}

    pend_regs = set()
    rtor = {}
    def reg_in(x):
      nonlocal pend_regs
      #print("reg_in", x, rtor[x], pend_regs)
      if x in pend_regs:
        #print("clear")
        ins.append('s_waitcnt lgkmcnt(0), vmcnt(0)')
        pend_regs.clear()
      return rtor[x]
    def reg_out(x):
      return rtor[x]
    for uop, out, vin, arg in asm:
      if uop == UOps.DEFINE_REGISTER:
        if arg[0][0] == dtypes.uint64 and arg[0][1]:
          # assuming these are scalar
          s_cnt += s_cnt%2  # aligned(2)
          for i in range(arg[2]):
            rtor[Register(f"%{arg[1]}{i}", *arg[0])] = f"s[{s_cnt}:{s_cnt+1}]"
            s_cnt += 2
        elif arg[0][0] in [dtypes.int32, dtypes.float32]:
          for i in range(arg[2]):
            if arg[0][1]:
              rtor[Register(f"%{arg[1]}{i}", *arg[0])] = f"s{s_cnt}"
              s_cnt += 1
            else:
              rtor[Register(f"%{arg[1]}{i}", *arg[0])] = f"v{v_cnt}"
              v_cnt += 1
        elif arg[0][0] == dtypes.bool and arg[0][1]:
          for i in range(arg[2]):
            rtor[Register(f"%{arg[1]}{i}", *arg[0])] = f"scc"
        else:
          raise NotImplementedError(arg)
      elif uop == UOps.SPECIAL:
        if arg.startswith('buf'):
          i = int(arg[3:])
          ins.append(f's_load_b64 {reg_out(out)}, s[0:1], {i*8}')
          pend_regs.add(out)
        elif arg.startswith('gid'):
          ins.append(f'v_mov_b32 {reg_out(out)}, s{2+int(arg[3])}')
      elif uop == UOps.CONST:
        ins.append(f"{'s_' if out.scalar else 'v_'}mov_b32 {reg_out(out)}, {arg}")
      elif uop == UOps.ALU:
        if arg == BinaryOps.CMPLT:
          ins.append(f"{'s_' if out.scalar else 'v_'}{alu[arg]}_{dtype_to_rdnatype[out.dtype]} {', '.join(reg_in(x) if x.__class__ is Register else str(x) for x in vin)}")
        elif arg == FusedOps.MULACC and out == vin[2]:
          ins.append(f"{'s_' if out.scalar else 'v_'}fmac_{dtype_to_rdnatype[out.dtype]} {reg_out(out)}, {', '.join(reg_in(x) if x.__class__ is Register else str(x) for x in vin[0:2])}")
        else:
          ins.append(f"{'s_' if out.scalar else 'v_'}{alu[arg]}_{dtype_to_rdnatype[out.dtype]}{'_i24' if arg == BinaryOps.MUL and out.dtype != dtypes.float32 and not out.scalar else ''} {reg_out(out)}, {', '.join(reg_in(x) if x.__class__ is Register else str(x) for x in vin)}")
      elif uop == UOps.LOAD:
        if out.scalar:
          # swap arg order
          ins.append(f's_load_b32 {reg_out(out)}, {reg_in(vin[0])}, {reg_in(arg[2])} offset:{arg[0]}')
        else:
          ins.append(f'global_load_b32 {reg_out(out)}, {reg_in(arg[2])}, {reg_in(vin[0])} offset:{arg[0]}')
        pend_regs.add(out)
      elif uop == UOps.STORE:
        ins.append(f'global_store_b32 {reg_in(arg[2])}, {reg_in(vin[1])}, {reg_in(vin[0])} offset:{arg[0]}')
      elif uop == UOps.LABEL:
        ins.append(f"{arg}:")
      elif uop == UOps.COND_BRANCH:
        ins.append(f"s_cbranch_scc{'1' if arg[1] else '0'} {arg[0]}")
      else:
        raise NotImplementedError(uop)

    ins += ['s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)', 's_endpgm', 's_code_end']
    return 'code', self.assemble(args, ins, v_cnt, s_cnt)

  def assemble(self, args, ins, v_cnt, s_cnt):
    kernel_desc = {'.amdhsa_group_segment_fixed_size': 0, '.amdhsa_private_segment_fixed_size': 0, '.amdhsa_kernarg_size': 0,
                   '.amdhsa_next_free_vgpr': v_cnt,   # this matters!
                   '.amdhsa_reserve_vcc': 0, '.amdhsa_reserve_xnack_mask': 0,
                   '.amdhsa_next_free_sgpr': s_cnt,
                   '.amdhsa_float_round_mode_32': 0, '.amdhsa_float_round_mode_16_64': 0, '.amdhsa_float_denorm_mode_32': 3, '.amdhsa_float_denorm_mode_16_64': 3, '.amdhsa_dx10_clamp': 1, '.amdhsa_ieee_mode': 1,
                   '.amdhsa_fp16_overflow': 0, '.amdhsa_workgroup_processor_mode': 1, '.amdhsa_memory_ordered': 1, '.amdhsa_forward_progress': 0, '.amdhsa_enable_private_segment': 0,
                   '.amdhsa_system_sgpr_workgroup_id_x': 1, '.amdhsa_system_sgpr_workgroup_id_y': 1, '.amdhsa_system_sgpr_workgroup_id_z': 1, '.amdhsa_system_sgpr_workgroup_info': 0, '.amdhsa_system_vgpr_workitem_id': 0,
                   '.amdhsa_exception_fp_ieee_invalid_op': 0, '.amdhsa_exception_fp_denorm_src': 0, '.amdhsa_exception_fp_ieee_div_zero': 0, '.amdhsa_exception_fp_ieee_overflow': 0, '.amdhsa_exception_fp_ieee_underflow': 0,
                   '.amdhsa_exception_fp_ieee_inexact': 0, '.amdhsa_exception_int_div_zero': 0, '.amdhsa_user_sgpr_dispatch_ptr': 0, '.amdhsa_user_sgpr_queue_ptr': 0, '.amdhsa_user_sgpr_kernarg_segment_ptr': 1,
                   '.amdhsa_user_sgpr_dispatch_id': 0, '.amdhsa_user_sgpr_private_segment_size': 0, '.amdhsa_wavefront_size32': 1, '.amdhsa_uses_dynamic_stack': 0}

    metadata = {'amdhsa.kernels': [{'.args': args,
                  '.group_segment_fixed_size': 0, '.kernarg_segment_align': 8, '.kernarg_segment_size': len(self.bufs)*8,
                  '.language': 'OpenCL C', '.language_version': [1, 2], '.max_flat_workgroup_size': 256,
                  '.name': 'code', '.private_segment_fixed_size': 0, '.sgpr_count': s_cnt, '.sgpr_spill_count': 0,
                  '.symbol': 'code.kd', '.uses_dynamic_stack': False, '.vgpr_count': v_cnt, '.vgpr_spill_count': 0,
                  '.wavefront_size': 32}],
                'amdhsa.target': 'amdgcn-amd-amdhsa--gfx1100', 'amdhsa.version': [1, 2]}

    code = boilerplate_start + "\n" + '\n'.join("%s %d" % x for x in kernel_desc.items()) + "\n" +  code_start + '\n'.join(ins) + "\n.amdgpu_metadata\n" + yaml.dump(metadata) + ".end_amdgpu_metadata"
    obj = early_exec(([ROCM_LLVM_PATH / "llvm-mc", '--arch=amdgcn', '--mcpu=gfx1100', '--triple=amdgcn-amd-amdhsa', '--filetype=obj', '-'], code.encode("utf-8")))
    asm = early_exec(([ROCM_LLVM_PATH / "ld.lld", "/dev/stdin", "-o", "/dev/stdout", "--pie"], obj))
    return asm
