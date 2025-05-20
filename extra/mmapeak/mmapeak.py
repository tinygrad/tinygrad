import pathlib
from tinygrad.device import Device
from tinygrad.runtime.ops_amd import AMDProgram, HIPCompiler
import time

NUM_WORKGROUPS = 96
WAVE_SIZE = 32
NUM_WAVES = 2
FLOPS_PER_MATMUL =  (16*16*16*2 + 16*16)
INTERNAL_LOOP  = 1_000_000_000

DEV = Device['AMD']
assemblyTemplate = (pathlib.Path(__file__).parent / "template.s").read_text()

def launchBenchmark(instruction, vgprCount):
  src = assemblyTemplate.replace("INSTRUCTION",
                                  instruction + " v[0:7], v[8:{}], v[{}:{}], v[{}:{}]"
                                  .format(8+vgprCount-1, 8+vgprCount, 8+2*vgprCount-1, 8+2*vgprCount, 8+2*vgprCount+8-1))
  lib = HIPCompiler("gfx1100").compile(src)
  fxn = AMDProgram(DEV, "matmul", lib)
  start = time.perf_counter()
  fxn(global_size=(NUM_WORKGROUPS,1,1), local_size=(WAVE_SIZE*NUM_WAVES,1,1), wait=True) #For some reason the returned time is very small after the first kernel execution
  end = time.perf_counter()
  elapsed = end-start
  FLOPs = FLOPS_PER_MATMUL * NUM_WAVES * NUM_WORKGROUPS * INTERNAL_LOOP
  print("{:<25} : {} T(FL)OPS".format(instruction, round(FLOPs/elapsed/10**12, 2)))

if __name__=="__main__":
  launchBenchmark("v_wmma_bf16_16x16x16_bf16", 8)
  launchBenchmark("v_wmma_f16_16x16x16_f16", 8)
  launchBenchmark("v_wmma_f32_16x16x16_bf16", 8)
  launchBenchmark("v_wmma_f32_16x16x16_f16", 8)
  launchBenchmark("v_wmma_i32_16x16x16_iu4", 2)
  launchBenchmark("v_wmma_i32_16x16x16_iu8", 4)