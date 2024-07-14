from tinygrad.tensor import Tensor


a = Tensor([1,2,3,4])
b = a.var()
print(b.numpy()) # --> 1.6666667

"""
run:
NOOPT=1 DEBUG=5 python temp/variance_calc_eg.py

output:
#include <metal_stdlib>
using namespace metal;
kernel void r_4(device float* data0, const device int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int acc0 = 0;
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int val0 = *(data1+ridx0);
    acc0 = (val0+acc0);
  }
  *(data0+0) = ((float)(acc0)*0.25f);
}

#include <metal_stdlib>
using namespace metal;
kernel void r_4n1(device float* data0, const device int* data1, const device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  float val0 = *(data2+0);
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int val1 = *(data1+ridx0);
    float alu0 = ((float)(val1)+(-val0));
    acc0 = ((alu0*alu0)+acc0);
  }
  *(data0+0) = (acc0*0.3333333432674408f);
}
"""