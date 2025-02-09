import numpy as np
from tinygrad import Tensor, GlobalCounters, Context
from tinygrad.engine.realize import lower_schedule, CompiledRunner
from tinygrad.device import CPUProgram
from dataclasses import replace
from keystone import Ks, KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN

reduce_asm = """
movi  v0.2d, #0000000000000000
add   x8, x1, #0x40
movi  v1.2d, #0000000000000000
mov   x9, #-0x20
movi  v2.2d, #0000000000000000
mov   w10, #0xffffe0
movi  v3.2d, #0000000000000000
loop:
ldp   q4, q5, [x8, #-0x40]
add   x9, x9, #0x20
cmp   x9, x10
fadd  v0.4s, v4.4s, v0.4s
ldp   q6, q7, [x8, #-0x20]
fadd  v1.4s, v5.4s, v1.4s
fadd  v2.4s, v6.4s, v2.4s
ldp   q4, q5, [x8]
fadd  v3.4s, v7.4s, v3.4s
fadd  v0.4s, v0.4s, v4.4s
ldp   q6, q7, [x8, #0x20]
fadd  v1.4s, v1.4s, v5.4s
add   x8, x8, #0x80
fadd  v2.4s, v2.4s, v6.4s
fadd  v3.4s, v3.4s, v7.4s
b.lo  loop
fadd  v2.4s, v2.4s, v3.4s
fadd  v0.4s, v1.4s, v0.4s
fadd  v0.4s, v2.4s, v0.4s
faddp v0.4s, v0.4s, v0.4s
faddp s0, v0.2s
str   s0, [x0]
ret
adr   x0, #0xe089b
"""

ks = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
arm_bytecode, _ = ks.asm(reduce_asm)
arm_bytecode = bytes(arm_bytecode)

reduce_src = """
// data1 is 16M inputs
typedef float float4 __attribute__((aligned(32),vector_size(16)));
void reduce(float* restrict data0, float* restrict data1) {
  float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 acc3 = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int ridx0 = 0; ridx0 < 16777216; ridx0+=32) {
    float4 val0 = *(float4*)((data1+(ridx0+0)));
    float4 val1 = *(float4*)((data1+(ridx0+4)));
    float4 val2 = *(float4*)((data1+(ridx0+8)));
    float4 val3 = *(float4*)((data1+(ridx0+12)));
    float4 val4 = *(float4*)((data1+(ridx0+16)));
    float4 val5 = *(float4*)((data1+(ridx0+20)));
    float4 val6 = *(float4*)((data1+(ridx0+24)));
    float4 val7 = *(float4*)((data1+(ridx0+28)));
    acc0 += val0;
    acc1 += val1;
    acc2 += val2;
    acc3 += val3;
    acc0 += val4;
    acc1 += val5;
    acc2 += val6;
    acc3 += val7;
  }
  float4 out = acc0+acc1+acc2+acc3;

  /*float4 out = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int ridx0 = 0; ridx0 < 16777216; ridx0+=4) {
    float4 val0 = *(float4*)((data1+(ridx0+0)));
    out += val0;
  }*/

  *(data0+0) = out[0]+out[1]+out[2]+out[3];
}
"""

if __name__ == "__main__":
  a = Tensor(np_array:=(np.random.default_rng().random((4096, 4096), dtype=np.float32)-0.5)).realize()
  with Context(SPLIT_REDUCEOP=0):
    # TODO: make it easy to alter the OptOps for a ScheduleItem
    GlobalCounters.reset()
    out = a.sum()
    sis = out.schedule()
    for i,ei in enumerate(lower_schedule(sis)):
      if i == 0:
        # change the source code
        prg_spec = ei.prg.p
        prg_spec = replace(prg_spec, name="reduce", src=reduce_src)
        prg = CompiledRunner(prg_spec)
        # change the assembly
        prg._prg = CPUProgram(prg_spec.name, arm_bytecode)
        ei = replace(ei, prg=prg)
      ei.run()
    print(out.item())
    np.testing.assert_allclose(out.item(), np_array.sum(), rtol=1e-4)
