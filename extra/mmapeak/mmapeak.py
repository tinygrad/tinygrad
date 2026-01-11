import os

# TODO: there is a timing bug without this
os.environ["AMD_AQL"] = "1"

from tinygrad.device import Device
from tinygrad.runtime.ops_amd import AMDProgram
from tinygrad.helpers import unwrap
from extra.amd_elf import build_hsaco

NUM_WORKGROUPS = 96
WAVE_SIZE = 32
NUM_WAVES = 2
FLOPS_PER_MATMUL = 16*16*16*2
INTERNAL_LOOP = 1_000_00
INSTRUCTIONS_PER_LOOP = 200
INST_ADD_I32 = None
KD:dict = {}

def launchBenchmark(instruction, vgprIndices, dense=True, accum=False, **kwargs):
  if accum:
    instructions = instruction(v[0:vgprIndices[0]], v[vgprIndices[1]:vgprIndices[2]], v[vgprIndices[1]:vgprIndices[2]], **kwargs)
  elif dense:
    instructions = instruction(v[0:vgprIndices[0]], v[vgprIndices[1]:vgprIndices[2]], v[vgprIndices[1]:vgprIndices[2]], 1)
  else:
    instructions = instruction(v[0:vgprIndices[0]], v[vgprIndices[1]:vgprIndices[2]], v[vgprIndices[3]:vgprIndices[4]], v[vgprIndices[5]])

  setup = [s_mov_b32(s[1], INTERNAL_LOOP), s_mov_b32(s[2], 0)]
  loop_body = [instructions]*INSTRUCTIONS_PER_LOOP + [unwrap(INST_ADD_I32)(s[1], 0xffff), s_cmp_lg_i32(s[1], s[2])]
  loop_size = sum(i.size() for i in loop_body)
  insts = setup + loop_body + [s_cbranch_scc1(-(loop_size + 4) // 4), s_endpgm()]
  lib = build_hsaco(insts, KD)
  fxn = AMDProgram(DEV, "matmul", lib)
  elapsed = min([fxn(global_size=(NUM_WORKGROUPS,1,1), local_size=(WAVE_SIZE*NUM_WAVES,1,1), wait=True) for _ in range(2)])
  FLOPs = FLOPS_PER_MATMUL * NUM_WAVES * NUM_WORKGROUPS * INTERNAL_LOOP * INSTRUCTIONS_PER_LOOP
  print(f"{(instructions.op.name.lower()):<29} : {FLOPs/elapsed/10**12:.2f} T(FL)OPS")

if __name__=="__main__":
  DEVICENUM = os.getenv("DEVICENUM", "0")
  try:
    DEV = Device['AMD:' + DEVICENUM]
  except:
    raise RuntimeError("Error while initiating AMD device")

  if DEV.arch in {'gfx1100', 'gfx1103', 'gfx1151'}:
    from extra.assembly.amd.autogen.rdna3.ins import *
    INST_ADD_I32 = s_add_i32
    KD = {"group_segment_fixed_size":0, "private_segment_fixed_size":0, "kernarg_size":0, "next_free_vgpr":32, "next_free_sgpr":8, "memory_ordered":1,
          "system_sgpr_workgroup_id_x":1, "system_sgpr_workgroup_id_y":0, "system_sgpr_workgroup_id_z":0, "wavefront_size32":1, "forward_progress":0,
          "user_sgpr_kernarg_segment_ptr":0, "user_sgpr_count":0, "workgroup_processor_mode":1, "uses_dynamic_stack":0}
    if DEV.arch == 'gfx1103': NUM_WORKGROUPS = 8
    if DEV.arch == 'gfx1151': NUM_WORKGROUPS = 32
    launchBenchmark(v_wmma_bf16_16x16x16_bf16, (7,8,15))
    launchBenchmark(v_wmma_f16_16x16x16_f16, (7,8,15))
    launchBenchmark(v_wmma_f32_16x16x16_bf16, (7,8,15))
    launchBenchmark(v_wmma_f32_16x16x16_f16, (7,8,15))
    launchBenchmark(v_wmma_i32_16x16x16_iu4, (7,8,9))
    launchBenchmark(v_wmma_i32_16x16x16_iu8, (7,8,11))
  elif DEV.arch in {'gfx1200', 'gfx1201'}:
    from extra.assembly.amd.autogen.rdna4.ins import *
    INST_ADD_I32 = s_addk_co_i32
    KD = {"next_free_vgpr":32, "next_free_sgpr":8, "wavefront_size32":1}
    NUM_WORKGROUPS = 64
    launchBenchmark(v_wmma_bf16_16x16x16_bf16, (3,4,7))
    launchBenchmark(v_wmma_f16_16x16x16_f16, (3,4,7))
    launchBenchmark(v_wmma_f32_16x16x16_bf16, (7,8,11))
    launchBenchmark(v_wmma_f32_16x16x16_f16, (7,8,11))
    launchBenchmark(v_wmma_i32_16x16x16_iu4, (7,8,8))
    launchBenchmark(v_wmma_i32_16x16x16_iu8, (7,8,9))
    launchBenchmark(v_wmma_f32_16x16x16_fp8_fp8, (7,8,9))
    launchBenchmark(v_wmma_f32_16x16x16_fp8_bf8, (7,8,9))
    launchBenchmark(v_wmma_f32_16x16x16_bf8_fp8, (7,8,9))
    launchBenchmark(v_wmma_f32_16x16x16_bf8_bf8, (7,8,9))
    FLOPS_PER_MATMUL = 16*16*32*2
    launchBenchmark(v_wmma_i32_16x16x32_iu4, (7,8,9))
    launchBenchmark(v_swmmac_f32_16x16x32_f16, (7,8,11,12,19,20), False)
    launchBenchmark(v_swmmac_f32_16x16x32_bf16, (7,8,11,12,19,20), False)
    launchBenchmark(v_swmmac_f16_16x16x32_f16, (3,4,7,8,15,16), False)
    launchBenchmark(v_swmmac_bf16_16x16x32_bf16, (3,4,7,8,15,16), False)
    launchBenchmark(v_swmmac_i32_16x16x32_iu8, (7,8,9,10,13,14), False)
    launchBenchmark(v_swmmac_i32_16x16x32_iu4, (7,8,8,9,10,11), False)
    launchBenchmark(v_swmmac_f32_16x16x32_fp8_fp8, (7,8,9,10,13,14), False)
    launchBenchmark(v_swmmac_f32_16x16x32_fp8_bf8, (7,8,9,10,13,14), False)
    launchBenchmark(v_swmmac_f32_16x16x32_bf8_fp8, (7,8,9,10,13,14), False)
    launchBenchmark(v_swmmac_f32_16x16x32_bf8_bf8, (7,8,9,10,13,14), False)
    FLOPS_PER_MATMUL = 16*16*64*2
    launchBenchmark(v_swmmac_i32_16x16x64_iu4, (7,8,9,10,13,14), False)
  elif DEV.arch == 'gfx950':
    from extra.assembly.amd.autogen.cdna.ins import *
    INST_ADD_I32 = s_add_i32
    KD = {"amdhsa_accum_offset":4}
    NUM_WORKGROUPS = 256
    WAVE_SIZE = 64
    NUM_WAVES = 4
    launchBenchmark(v_mfma_f32_16x16x16_f16, (3,0,1), accum=True)
    launchBenchmark(v_mfma_f32_16x16x16_bf16, (3,0,1), accum=True)
    FLOPS_PER_MATMUL = 16*16*32*2
    launchBenchmark(v_mfma_f32_16x16x32_f16, (3,0,3), accum=True)
    launchBenchmark(v_mfma_f32_16x16x32_bf16, (3,0,3), accum=True)
    FLOPS_PER_MATMUL = 16*16*128*2
    launchBenchmark(v_mfma_f32_16x16x128_f8f6f4, (3,0,7), accum=True) # fp8
    launchBenchmark(v_mfma_f32_16x16x128_f8f6f4, (3,0,5), accum=True, cbsz=2, blgp=2) # fp6
    launchBenchmark(v_mfma_f32_16x16x128_f8f6f4, (3,0,3), accum=True, cbsz=2, blgp=2) # fp4
  else:
    raise RuntimeError(f"arch {DEV.arch} not supported.")
