#!/usr/bin/env python3
"""Disassemble a compiler-generated 64-bit global-pointer increment."""
import struct

from tinygrad import Device
from extra.gemm.ir3asm import disasm

SRC = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void stores(__global half *O) {
  int t=get_global_id(0); __global half *p=O+t*4;
  vstore4((half4)(1),0,p); vstore4((half4)(2),0,p+128);
}"""

lib = Device["QCOM"].compiler.compile(SRC)
off, size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
print(disasm(lib[off:off+size]))
