#!/usr/bin/env python3
"""Inspect uint4 image-load to half8 register layout on A630."""
import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from extra.gemm.ir3asm import NOP, disasm, get_envelope, inject


SRC = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void probe(read_only image2d_t X,__global half *O) {
  int i=get_global_id(0); half8 h=as_half8(read_imageui(X,smp,(int2)(i,0))); vstore8(h,0,O+i*8);
}"""


def main() -> None:
  dev=Device["QCOM"]; lib,off,size,ro=get_envelope(dev,SRC); print(disasm(bytes(lib[off:off+size])))
  values=np.arange(64*8,dtype=np.float16).view(np.uint32).reshape(1,64,4)
  src=Buffer("QCOM",values.size,dtypes.uint32).allocate(); out=Buffer("QCOM",128*8,dtypes.half).allocate()
  src.copyin(memoryview(values).cast("B"))
  prg=dev.runtime("probe",bytes(lib),buf_dtypes=[((0,dtypes.uint32,(1,64,4)),),((1,dtypes.half,None),)])
  prg(src._buf,out._buf,global_size=(1,1,1),local_size=(128,1,1),wait=True)
  got=np.empty(128*8,np.float16); out.copyout(memoryview(got).cast("B")); print("head",got[:16].tolist())
  shader=bytearray(lib[off:off+size])
  for i in range(13,21): shader[i*8:(i+1)*8]=NOP()
  for mode in (None, True, False):
    patched=inject(lib,off,size,ro,bytes(shader),fregs=2,hregs=2,mergedregs=mode)
    out.copyin(memoryview(np.zeros(128*8,np.float16)).cast("B"))
    hand=dev.runtime("probe",patched,buf_dtypes=[((0,dtypes.uint32,(1,64,4)),),((1,dtypes.half,None),)])
    hand(src._buf,out._buf,global_size=(1,1,1),local_size=(128,1,1),wait=True)
    out.copyout(memoryview(got).cast("B")); print("direct_alias",mode,got[:16].tolist())


if __name__=="__main__": main()
