from typing import Any, List, cast
from tinygrad.device import Device
from tinygrad.runtime.ops_metal import MetalDevice, MetalCompiler, MetalAllocator, MetalProgram

def verify(code, global_size, local_size):
  device = cast(MetalDevice, Device[Device.DEFAULT])
  lib = MetalCompiler(device).compile(code)
  prg = MetalProgram(device, "test", lib)
  allocator = MetalAllocator(device)
  def to_bytes(x): return bytearray([byte for value in x for byte in value.to_bytes(4, 'little', signed=True)])
  def alloc_data(x:List[int]):
    data:Any = to_bytes(x)
    d = allocator._alloc(len(data))
    allocator.copyin(d, data)
    return d
  a = alloc_data([1,2,3,4])
  output = allocator.alloc(4)
  prg(output, a, global_size=global_size, local_size=local_size)
  return allocator.as_buffer(output).cast("I").tolist()

code = """
#include <metal_stdlib>
using namespace metal;
kernel void test(device int* data0, device int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int acc0 = 0;
  for (int ridx0 = 0; ridx0 < 5; ridx0++) {
    int val0 = *(data1+ridx0);
    acc0 = val0+acc0;
  }
  *(data0+0) = acc0;
}
"""
assert verify(code, (1,1,1), (1,1,1)) == [10]

code = """
#include <metal_stdlib>
using namespace metal;
kernel void test(device int* data0, device int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup int temp[4];
  int lidx0 = lid.x; /* 2 */
  int acc0 = 0;
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    int alu0 = ridx0+lidx0*2;
    int val0 = *(data1+alu0);
    acc0 = val0+acc0; // 1+2, 3+4
  }

  *(temp+lidx0) = acc0; // [3, 7]

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lidx0 == 0) {
    int acc1 = 0;
    for (int ridx1=0; ridx1<2; ridx1++) {
      int val1 = *(temp+ridx1); // [3, 7]
      acc1 = val1+acc1; // 3+7
    }
    *(data0+0) = acc1;
  }
}
"""
assert verify(code, (1,1,1), (2,1,1)) == [10]
