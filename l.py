from tinygrad.ops import Compiled, Device, LoadOps
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.runtime.ops_metal import MetalProgram
from tinygrad.features.search import bufs_from_lin
from tinygrad.tensor import Tensor


device: Compiled = Device[Device.DEFAULT]
si = [si for si in Tensor([1,2,3]).add(42).lazydata.schedule() if si.ast.op not in LoadOps][0]
lin = Linearizer(si.ast, device.linearizer_opts)
prg = device.to_program(lin)
rawbufs = bufs_from_lin(lin)
tm = prg(rawbufs=rawbufs, force_wait=True)
print(rawbufs[1].toCPU(), rawbufs[0].toCPU())

code = """
#include <metal_stdlib>
using namespace metal;
kernel void E_3(device float* data0, const device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 3 */
  float val0 = *(data1+gidx0);
  *(data0+gidx0) = (val0+45.0f);
}
"""
prg = MetalProgram(lin.function_name, code)
tm = prg(lin.global_size, lin.local_size, *rawbufs, wait=True)
print(rawbufs[1].toCPU(), rawbufs[0].toCPU())
