import functools
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage

class MetalLanguage(CStyleLanguage):
  kernel_prefix = "#include <metal_stdlib>\nusing namespace metal;\nkernel "
  buffer_prefix = "device "
  smem_prefix = "threadgroup "
  arg_int_prefix = "constant int&"
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
  float4 = "float4"
  uses_ptr_arithmetic=True
  gid = [f"gid.{chr(120+i)}" for i in range(3)]
  lid = [f"lid.{chr(120+i)}" for i in range(3)]
  extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']

  def render_cast(self, x, var_dtype, bitcast=False) -> str:
    return super().render_cast(x, var_dtype, bitcast)
    if not bitcast: return super().render_cast(x, var_dtype, bitcast)
    if len(x) == 1: return f"as_type<{var_dtype.name}>({x[0]})"
    # return ",".join([f"as_type<{var_dtype.name}>({v})" for v in x])

MetalRenderer = functools.partial(uops_to_cstyle, MetalLanguage())
