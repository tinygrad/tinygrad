from tinygrad.runtime.metal import CLBuffer, CLProgram

"""
def benchmark(prog):
  e = prog()
  e.waitUntilCompleted()
  return (e.GPUEndTime() - e.GPUStartTime())*1e9
def mb(prog, N=10): return min([benchmark(prog) for _ in range(N)])

N = 2048
a = CLBuffer(N*N*4)
b = CLBuffer(N*N*4)
c = CLBuffer(N*N*4)
"""

prog = CLProgram("test", f"""
using namespace metal;
kernel void test(device float *a, device const float *data1, device const float *data2, uint3 gid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
  int i = gid.x;
  float4 x = ((device float4*)data1)[i];
  float4 y = ((device float4*)data2)[i];
  ((device float4*)a)[i] += x*y;
  //a[i] = dot(x,y);

  /*float x = data1[i];
  float4 y = ((device float4*)data2)[i];
  ((device float4*)a)[i] = x*y;*/
}}""")
#tm = mb(lambda: prog([N*N//(2*4*4)], [4*32], a._cl, b._cl, c._cl))
