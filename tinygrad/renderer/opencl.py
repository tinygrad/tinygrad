import functools
from tinygrad.helpers import dtypes
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage

type_map = { dtypes.uint8: "uchar", dtypes.uint32: "uint", dtypes.uint64: "ulong" }
class OpenCLLanguage(CStyleLanguage):
  kernel_prefix = "__kernel "
  buffer_prefix = "__global "
  smem_align = "__attribute__ ((aligned (16))) "
  smem_prefix = "__local "
  arg_int_prefix = "const int"
  half_prekernel = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable"
  barrier = "barrier(CLK_LOCAL_MEM_FENCE);"
  float4 = "(float4)"
  gid = [f'get_group_id({i})' for i in range(3)]
  lid = [f'get_local_id({i})' for i in range(3)]
  uses_vload=True

OpenCLRenderer = functools.partial(uops_to_cstyle, OpenCLLanguage())
