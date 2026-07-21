#!/usr/bin/env python3
import struct
from tinygrad import Device, dtypes
from tinygrad.device import Buffer

src = r"""const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(32,1,1)))
__kernel void k(read_only image2d_t A,__global uint *C) {
  int i=get_global_id(0); uint4 v=read_imageui(A,smp,(int2)(i,0)); vstore4(v,0,C+i*4);
}"""
dev=Device["QCOM"]
lib=dev.compiler.compile_cached(src)
a=Buffer("QCOM",128,dtypes.uint32).allocate()
c=Buffer("QCOM",128,dtypes.uint32).allocate()
vals=list(range(128))
a.copyin(memoryview(bytearray(struct.pack("<128I",*vals))))
c.copyin(memoryview(bytearray(512)))
p=dev.runtime("k",lib,buf_dtypes=[((0,dtypes.uint32,(1,32,4)),),((0,dtypes.uint32,None),)])
print(p(a._buf,c._buf,global_size=(32,1,1),local_size=(32,1,1),wait=True))
out=bytearray(512)
c.copyout(memoryview(out))
got=struct.unpack("<128I",out)
print(got[:16],got==tuple(vals))
