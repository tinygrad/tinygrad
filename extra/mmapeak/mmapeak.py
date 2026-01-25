import os

# TODO: there is a timing bug without this
os.environ["AMD_AQL"] = "1"

from tinygrad.device import Device
from tinygrad.runtime.support.compiler_amd import HIPCompiler

from extra.assembly.amd.dsl import s, v, Inst
from extra.amdgpu_elf import pack_hsaco

NUM_WORKGROUPS = 96
WAVE_SIZE = 32
NUM_WAVES = 2
FLOPS_PER_MATMUL = 16*16*16*2
INTERNAL_LOOP = 1_000_00
INSTRUCTIONS_PER_LOOP = 200

# arch specific instruction for S_LOOP_N -= 1
S_LOOP_N = s[1]
INST_LOOP_STEP:Inst|None = None

def launchBenchmark(instruction, vgprIndices, dense=True, accum=False, **kwargs):
  if accum:
    instructions = instruction(v[0:vgprIndices[0]], v[vgprIndices[1]:vgprIndices[2]], v[vgprIndices[1]:vgprIndices[2]], 1, **kwargs)
  elif dense:
    instructions = instruction(v[0:vgprIndices[0]], v[vgprIndices[1]:vgprIndices[2]], v[vgprIndices[1]:vgprIndices[2]], 1)
  else:
    instructions = instruction(v[0:vgprIndices[0]], v[vgprIndices[1]:vgprIndices[2]], v[vgprIndices[3]:vgprIndices[4]], v[vgprIndices[5]])
  insts = [s_mov_b32(S_LOOP_N, INTERNAL_LOOP)]
  loop_body = [instructions]*INSTRUCTIONS_PER_LOOP + [INST_LOOP_STEP, s_cmp_lg_i32(S_LOOP_N, 0)]
  insts += loop_body + [s_cbranch_scc1((-(sum(i.size() for i in loop_body) + 4) // 4) & 0xff), s_endpgm()]
  lib = pack_hsaco(b"".join(i.to_bytes() for i in insts), KD)
  fxn = DEV.runtime("matmul", lib)
  elapsed = min([fxn(global_size=(NUM_WORKGROUPS,1,1), local_size=(WAVE_SIZE*NUM_WAVES,1,1), wait=True) for _ in range(2)])
  FLOPs = FLOPS_PER_MATMUL * NUM_WAVES * NUM_WORKGROUPS * INTERNAL_LOOP * INSTRUCTIONS_PER_LOOP
  print(f"{(instructions.op_name.lower()):<29} : {FLOPs/elapsed/10**12:.2f} T(FL)OPS")

if __name__=="__main__":
  DEV = Device[Device.DEFAULT]
  arch = DEV.renderer.arch

  if arch in {'gfx1100', 'gfx1103', 'gfx1151'}:
    from extra.assembly.amd.autogen.rdna3.ins import *
    INST_LOOP_STEP = s_add_i32(S_LOOP_N, S_LOOP_N, -1)
    KD = {"next_free_vgpr":32, "next_free_sgpr":1, "wavefront_size32":1}
    if arch == 'gfx1103': NUM_WORKGROUPS = 8
    if arch == 'gfx1151': NUM_WORKGROUPS = 32
    launchBenchmark(v_wmma_bf16_16x16x16_bf16, (7,8,15))
    launchBenchmark(v_wmma_f16_16x16x16_f16, (7,8,15))
    launchBenchmark(v_wmma_f32_16x16x16_bf16, (7,8,15))
    launchBenchmark(v_wmma_f32_16x16x16_f16, (7,8,15))
    launchBenchmark(v_wmma_i32_16x16x16_iu4, (7,8,9))
    launchBenchmark(v_wmma_i32_16x16x16_iu8, (7,8,11))
  elif arch in {'gfx1200', 'gfx1201'}:
    from extra.assembly.amd.autogen.rdna4.ins import *
    INST_LOOP_STEP = s_addk_co_i32(S_LOOP_N, 0xffff)
    KD = {"next_free_vgpr":32, "next_free_sgpr":1, "wavefront_size32":1}
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
  elif arch == 'gfx950':
    from extra.assembly.amd.autogen.cdna.ins import *
    INST_LOOP_STEP = s_add_i32(S_LOOP_N, S_LOOP_N, -1)
    KD = {"amdhsa_accum_offset":4, "next_free_vgpr":32, "next_free_sgpr":1}
    DIRECTIVE = ".amdhsa_accum_offset 4"
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
    launchBenchmark(v_mfma_f32_16x16x128_f8f6f4, (3,0,5), accum=True, neg=2, neg_hi=2) # fp6
    launchBenchmark(v_mfma_f32_16x16x128_f8f6f4, (3,0,3), accum=True, neg=4, neg_hi=4) # fp4
  else:
    raise RuntimeError(f"arch {arch} not supported.")
