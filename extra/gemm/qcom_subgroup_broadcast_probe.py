#!/usr/bin/env python3
"""Compile and validate A630 subgroup broadcast before using it in GEMM."""
import os, struct

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.ir3asm import disasm, get_envelope, inject, QUAD_BRCST
from extra.gemm.qcom_8x4_gemm import buf_copyin, buf_copyout


SRC = r"""#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void broadcast(__global uint *O, __global uint *X) {
  uint i=get_global_id(0);
  uint v=X[i];
  O[i]=qcom_sub_group_shuffle_xor(v,1,CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM,999u);
}"""


def main() -> None:
  dev=Device["QCOM"]
  src = SRC
  if (api := os.getenv("API", "xor")) != "xor":
    call = f"qcom_sub_group_shuffle_{api}(v,{os.getenv('SELECTOR', 'i&31')},CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM,999u)"
    src = src.replace("qcom_sub_group_shuffle_xor(v,1,CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM,999u)", call)
  lib0, off, size, reg_off = get_envelope(dev, src)
  lib=bytearray(lib0)
  if int(os.getenv("QBC", "0")):
    # Preserve the compiler's proven load/store and dependency schedule, but
    # replace its shuffle source setup and operation with quad broadcast lane 0.
    # The compiler put 999 in r0.w; QBC uses its low two bits, selecting lane 3.
    if int(os.getenv("QBC_ZERO", "0")):
      from extra.gemm.ir3asm import MOV_F32, MOV_S32, NOP
      lib[off+11*8:off+12*8] = MOV_S32('r0.w', int(os.getenv("QBC_SELECTOR", "0")))
      lib[off+12*8:off+13*8] = NOP(rpt=5)
      lib[off+13*8:off+14*8] = QUAD_BRCST('r0.y', 'r0.z', 'r0.w', typ=3, sy=True)
      lib[off+14*8:off+15*8] = MOV_F32('r0.w', 'r0.y')
    elif int(os.getenv("QBC_NO_WAW", "0")):
      from extra.gemm.ir3asm import NOP
      lib[off+11*8:off+12*8] = NOP()
      lib[off+13*8:off+14*8] = QUAD_BRCST('r0.w', 'r0.z', 'r0.y', typ=3, sy=True)
    elif int(os.getenv("QBC_DST1", "0")):
      from extra.gemm.ir3asm import MOV_F32, NOP
      lib[off+11*8:off+12*8] = QUAD_BRCST('r1.x', 'r0.z', 'r0.y', typ=3, sy=True)
      lib[off+12*8:off+13*8] = NOP(rpt=5)
      lib[off+13*8:off+14*8] = MOV_F32('r0.w', 'r1.x')
    elif int(os.getenv("QBC_KNOWN", "0")):
      lib[off+13*8:off+14*8] = bytes.fromhex("010400000731e0b7")
      # Preserve the store's r0.w source while avoiding an overlapping qbc destination.
      lib[off+14*8:off+15*8] = bytes.fromhex("0700000003c00c20")
    elif int(os.getenv("QBC_CONST_IDX", "0")):
      # r0.y is the compiler's zero-valued high-address carry here.
      lib[off+13*8:off+14*8] = QUAD_BRCST('r0.w', 'r0.z', 'r0.y', typ=3, sy=True)
      if int(os.getenv("QBC_WAIT", "0")):
        from extra.gemm.ir3asm import NOP
        lib[off+14*8:off+15*8] = NOP(rpt=int(os.getenv("QBC_WAIT")))
    else:
      lib[off+13*8:off+14*8] = (QUAD_BRCST('r0.w', 'r0.w', 'r0.z', typ=3, sy=True) if int(os.getenv("QBC_SWAP", "0")) else
                                    QUAD_BRCST('r0.w', 'r0.z', 'r0.w', typ=3, sy=True))
  if int(os.getenv("MERGED0", "0")):
    lib = bytearray(inject(lib, off, size, reg_off, bytes(lib[off:off+size]), fregs=2, hregs=0, mergedregs=False))
  print(disasm(lib[off:off+size]))
  n=256
  x=np.arange(n,dtype=np.uint32)
  xb,ob=Buffer("QCOM",n,dtypes.uint).allocate(),Buffer("QCOM",n,dtypes.uint).allocate()
  buf_copyin(xb, memoryview(x).cast("B"))
  prg=dev.runtime("broadcast",lib,buf_dtypes=[((0,dtypes.uint,None),),((1,dtypes.uint,None),)])
  times=[prg(ob._buf,xb._buf,global_size=(2,1,1),local_size=(128,1,1),wait=True) for _ in range(10)]
  got=np.empty(n,np.uint32); buf_copyout(ob, memoryview(got).cast("B"))
  print("min_us",min(times)*1e6)
  print("blocks",[np.unique(got[i:i+32]).tolist() for i in range(0,n,32)])
  expected = ((np.arange(n, dtype=np.uint32)//4)*4+3 if int(os.getenv("QBC", "0")) else
              (np.arange(n, dtype=np.uint32)//32)*32 if api == "up" else np.arange(n, dtype=np.uint32)^1)
  print("exact", bool(np.array_equal(got, expected)), "head", got[:32].tolist())
  if int(os.getenv("MAP", "0")):
    print("map16", [np.unique(got[i:i+16]).tolist() for i in range(0, n, 16)])


if __name__ == "__main__": main()
