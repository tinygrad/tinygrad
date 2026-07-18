#!/usr/bin/env python3
"""Disassemble the proprietary compiler's consecutive image-read schedule."""
import struct

from tinygrad import Device
from extra.gemm.ir3asm import disasm


SRC = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void reads(read_only image2d_t X,__global half *O) {
  int x=get_global_id(0), y=get_group_id(1)*4;
  half4 a=read_imageh(X,smp,(int2)(x,y+0));
  half4 b=read_imageh(X,smp,(int2)(x,y+1));
  half4 c=read_imageh(X,smp,(int2)(x,y+2));
  half4 d=read_imageh(X,smp,(int2)(x,y+3));
  vstore4(a+b+c+d,0,O+x*4+y*4096);
}"""


def main() -> None:
  lib=Device["QCOM"].compiler.compile(SRC)
  off,size=struct.unpack_from("<I",lib,0xc0)[0],struct.unpack_from("<I",lib,0x100)[0]
  print(disasm(lib[off:off+size]))


if __name__ == "__main__": main()
