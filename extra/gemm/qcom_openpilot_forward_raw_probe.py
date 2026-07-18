#!/usr/bin/env python3
"""Compile/disassemble the raw-accumulator variant of the vision forward tile."""
import struct

from tinygrad import Device
from extra.gemm.ir3asm import disasm
from extra.gemm.qcom_openpilot_forward_tile8 import SOURCE

RAW_TAIL = r"""  int r=row0;
#define STORE(v) { int m=r&31,p=r>>5; write_imageh(O,(int2)(n0+p*192,m),v.lo); write_imageh(O,(int2)(n1+p*192,m),v.hi); r++; }
  STORE(c0); STORE(c1); STORE(c2); STORE(c3); STORE(c4); STORE(c5); STORE(c6); STORE(c7);
}
"""
RAW_SOURCE = SOURCE[:SOURCE.index("  float4 b0=")] + RAW_TAIL

if __name__ == "__main__":
  lib = Device["QCOM"].compiler.compile_cached(RAW_SOURCE)
  image_off, image_size = struct.unpack_from("<I", lib, 0xc0)[0], struct.unpack_from("<I", lib, 0x100)[0]
  lines = [x for x in disasm(lib[image_off:image_off+image_size]).splitlines() if not x.rstrip().endswith(":")]
  print("COUNT", len(lines))
  for index, line in enumerate(lines): print(f"{index}: {line}")
