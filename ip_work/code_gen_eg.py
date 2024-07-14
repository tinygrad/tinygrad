from tinygrad import dtype
from tinygrad.dtype import dtypes
from tinygrad.codegen.uops import UOp, UOpGraph, UOps
from tinygrad.renderer.cstyle import MetalRenderer


uop_0 = UOp(
    op = UOps.DEFINE_GLOBAL,
    dtype = dtype.PtrDType(dtypes.float16),
    src = (),
    arg = (0, 'data0', True)
    # arg = (0, True)
)

uops_list = [uop_0]
uops = UOpGraph(uops_list)

metal_renderer = MetalRenderer()

metal_renderer_output = metal_renderer.render("test", uops)

# print(f"{output=}")
print("metal_renderer_output")
print(metal_renderer_output)

"""
input:
output = metal_renderer.render("test", [])

output:
#include <metal_stdlib>
using namespace metal;
kernel void test(uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {

}
"""
