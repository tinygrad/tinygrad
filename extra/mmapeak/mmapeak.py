import pathlib
from tinygrad.device import Device
from tinygrad.runtime.ops_amd import AMDProgram, HIPCompiler
import time
import os

NUM_WORKGROUPS = 96
WAVE_SIZE = 32
NUM_WAVES = 2
FLOPS_PER_MATMUL =  16*16*16*2 + 16*16
INTERNAL_LOOP  = 1_000_000
INSTRUCTIONS_PER_LOOP = 1_000

assemblyTemplate = (pathlib.Path(__file__).parent / "template.s").read_text()

def launchBenchmark(instruction, vgprIndices):
  src = assemblyTemplate.replace("INSTRUCTION",
                                  "{} v[0:{}], v[{}:{}], v[{}:{}], 1\n"
                                  .format(instruction, vgprIndices[0], vgprIndices[1], vgprIndices[2],
                                           vgprIndices[1], vgprIndices[2]) * INSTRUCTIONS_PER_LOOP)
  lib = COMPILER.compile(src)
  fxn = AMDProgram(DEV, "matmul", lib)
  start = time.perf_counter()
  fxn(global_size=(NUM_WORKGROUPS,1,1), local_size=(WAVE_SIZE*NUM_WAVES,1,1), wait=True) #For some reason the returned time is very small after the first kernel execution
  end = time.perf_counter()
  elapsed = end-start
  FLOPs = FLOPS_PER_MATMUL * NUM_WAVES * NUM_WORKGROUPS * INTERNAL_LOOP * INSTRUCTIONS_PER_LOOP
  print("{:<27} : {} T(FL)OPS".format(instruction, round(FLOPs/elapsed/10**12, 2)))

if __name__=="__main__":
  DEVICENUM = os.getenv("DEVICENUM", "0")
  try:
    DEV = Device['AMD:' + DEVICENUM]
  except:
    raise RuntimeError("Error while initiating AMD device")

  if (ARCH := DEV.arch) not in ['gfx1100', 'gfx1201']:
    raise RuntimeError("only gfx1100 and gfx1201 supported")
  COMPILER = HIPCompiler(ARCH)

  if ARCH == 'gfx1100':
    launchBenchmark("v_wmma_bf16_16x16x16_bf16", (7,8,15))
    launchBenchmark("v_wmma_f16_16x16x16_f16", (7,8,15))
    launchBenchmark("v_wmma_f32_16x16x16_bf16", (7,8,15))
    launchBenchmark("v_wmma_f32_16x16x16_f16", (7,8,15))
    launchBenchmark("v_wmma_i32_16x16x16_iu4", (7,8,9))
    launchBenchmark("v_wmma_i32_16x16x16_iu8", (7,8,11))
  if ARCH == 'gfx1201':
    NUM_WORKGROUPS = 64
    launchBenchmark("v_wmma_bf16_16x16x16_bf16", (3,4,7))
    launchBenchmark("v_wmma_f16_16x16x16_f16", (3,4,7))
    launchBenchmark("v_wmma_f32_16x16x16_bf16", (7,8,11))
    launchBenchmark("v_wmma_f32_16x16x16_f16", (7,8,11))
    launchBenchmark("v_wmma_i32_16x16x16_iu4", (7,8,8))
    launchBenchmark("v_wmma_i32_16x16x16_iu8", (7,8,9))
    launchBenchmark("v_wmma_f32_16X16X16_fp8_fp8", (7,8,9))
    launchBenchmark("v_wmma_f32_16X16X16_fp8_bf8", (7,8,9))
    launchBenchmark("v_wmma_f32_16X16X16_bf8_fp8", (7,8,9))
    launchBenchmark("v_wmma_f32_16X16X16_bf8_bf8", (7,8,9))