#!/usr/bin/env python3
"""Show the exact values returned by Qualcomm vector image builtins."""
import os
import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer


SRC = r"""#pragma OPENCL EXTENSION cl_khr_fp16 : enable
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST;
__attribute__((reqd_work_group_size(1,1,1)))
__kernel void probe(read_only image2d_t X, __global half *O) {
  half4 v0=qcom_read_imageh_4x1(X,smp,(float2)(5,7),0);
  half4 v1=qcom_read_imageh_4x1(X,smp,(float2)(5,7),1);
  half4 v2=qcom_read_imageh_4x1(X,smp,(float2)(5,7),2);
  half4 v3=qcom_read_imageh_4x1(X,smp,(float2)(5,7),3);
  vstore4(v0,0,O+0); vstore4(v1,0,O+4); vstore4(v2,0,O+8); vstore4(v3,0,O+12);
}"""


def main() -> None:
  h, w = 16, 32
  values = np.empty((h, w, 4), np.float16)
  for y in range(h):
    for x in range(w):
      for c in range(4): values[y, x, c] = y*100+x*4+c
  src = Buffer("QCOM", values.size, dtypes.half).allocate()
  out = Buffer("QCOM", 16, dtypes.half).allocate()
  src.copyin(memoryview(values).cast("B"))
  src_text = SRC.replace("qcom_read_imageh_4x1", os.getenv("QCOM_IMAGE_FN", "qcom_read_imageh_4x1"))
  prg = Device["QCOM"].runtime("probe", Device["QCOM"].compiler.compile(src_text),
    buf_dtypes=[((0, dtypes.half, values.shape),), ((1, dtypes.half, None),)])
  prg(src._buf, out._buf, global_size=(1, 1, 1), local_size=(1, 1, 1), wait=True)
  got = np.empty(16, np.float16); out.copyout(memoryview(got).cast("B"))
  print("got", got.reshape(4, 4).tolist())
  print("texels_x5_to_x8", values[7, 5:9].tolist())


if __name__ == "__main__": main()
