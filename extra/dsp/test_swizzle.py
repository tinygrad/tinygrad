import numpy as np
from tinygrad import Tensor, Device
from tinygrad.codegen.kernel import Kernel
from dataclasses import replace
from tinygrad.engine.realize import CompiledRunner, ExecItem

replace_src = """
typedef unsigned char unsigned_char128 __attribute__((aligned(128),vector_size(128)));
typedef unsigned char unsigned_char256 __attribute__((aligned(256),vector_size(256)));
__attribute__((noinline)) void E_512(unsigned char* restrict __attribute__((align_value(128))) data0, unsigned char* restrict __attribute__((align_value(128))) data1) {
  unsigned_char128 x0 = *(unsigned_char128*)(data1+0);
  unsigned_char128 x1 = *(unsigned_char128*)(data1+128);
  unsigned_char128 x2 = *(unsigned_char128*)(data1+255);
  unsigned_char128 x3 = *(unsigned_char128*)(data1+383);

  union V256 {
    unsigned_char256 vec256;
    struct {
      unsigned_char128 lo128;
      unsigned_char128 hi128;
    };
  };

  union V256 ss01;
  // ss01.lo128 = (x0[0], x1[0], x0[2], x1[2], x0[4], x1[4], ...)
  // ss01.hi128 = (x0[1], x1[1], x0[3], x1[3], x0[5], x1[5], ...)
  ss01.vec256 = __builtin_HEXAGON_V6_vshufoeb_128B(x1, x0);

  union V256 ss23;
  // ss23.lo128 = (x2[0], x3[0], x2[2], x3[2], x2[4], x3[4], ...)
  // ss23.hi128 = (x2[1], x3[1], x2[3], x3[3], x2[5], x3[5], ...)
  ss23.vec256 = __builtin_HEXAGON_V6_vshufoeb_128B(x3, x2);

  union V256 sslo;
  // sslo.lo128 = (x0[0], x1[0], x2[0], x3[0], x0[4], x1[4], ...)
  // sslo.hi128 = (x0[2], x1[2], x2[2], x3[2], x0[6], x1[6], ...)
  sslo.vec256 = __builtin_HEXAGON_V6_vdealvdd_128B(ss23.lo128, ss01.lo128, 2);

  *(unsigned_char128*)(data0+0) = sslo.hi128;
  //*(unsigned_char128*)(data0+256) = ss13.lo128;
  //*(unsigned_char128*)(data0+128) = x1;
}
"""

if __name__ == "__main__":
  Device.DEFAULT = "DSP"
  aa = Tensor.arange(512, dtype='uint8').realize()
  out = aa+1
  si = out.schedule()[-1]
  k = Kernel(si.ast, opts=Device[Device.DEFAULT].renderer)
  prg = k.to_program()
  prg = replace(prg, src=replace_src + "/* DSP boilerplate */" + prg.src.split("/* DSP boilerplate */")[1])
  ei = ExecItem(CompiledRunner(prg), [x.ensure_allocated() for x in si.bufs], si.metadata)
  ei.run(wait=True)
  print(np.array(si.bufs[0].as_buffer()))
