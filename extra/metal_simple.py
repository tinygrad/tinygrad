from tinygrad.runtime.metal import CLBuffer, CLProgram

def benchmark(prog):
  e = prog()
  e.waitUntilCompleted()
  return (e.GPUEndTime() - e.GPUStartTime())*1e9
def mb(prog, N=10): return min([benchmark(prog) for _ in range(N)])

N = 2048
a = CLBuffer(N*N*4)
b = CLBuffer(N*N*4)
c = CLBuffer(N*N*4)

prog = CLProgram("test", f"""
using namespace metal;
kernel void test(device float4 *a, device const float4 *data1, device const float4 *data2, uint3 gid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
  float4 r0 = data1[lid.x+0];
  float4 r1 = data1[lid.x+1];
  float4 r2 = data1[lid.x+2];
  float4 r3 = data1[lid.x+3];
  float4 b0 = data2[lid.x+0];
  float4 b1 = data2[lid.x+1];
  float4 b2 = data2[lid.x+2];
  float4 b3 = data2[lid.x+3];
  float4 acc[16];
  for (uint i = 0; i < 256; i++) {{
    acc[0] += r0 * b0;
    acc[1] += r0 * b1;
    acc[2] += r0 * b2;
    acc[3] += r0 * b3;
    acc[4] += r1 * b0;
    acc[5] += r1 * b1;
    acc[6] += r1 * b2;
    acc[7] += r1 * b3;
    acc[8] += r2 * b0;
    acc[9] += r2 * b1;
    acc[10] += r2 * b2;
    acc[11] += r2 * b3;
    acc[12] += r3 * b0;
    acc[13] += r3 * b1;
    acc[14] += r3 * b2;
    acc[15] += r3 * b3;
  }}
  float4 facc = 0;
  for (uint i = 0; i < 16 ; i++) facc += acc[i];
  a[gid.x] = facc;
}}""")
#      fma  float4   statements    loop
FLOPS = 2 *   4    *   16       *  256
sz = 512*512
tm = mb(lambda: prog([sz,1,1], [32,1,1], a._cl, b._cl, c._cl))
print(f"{sz:10d} {tm*1e-3:9.2f} us {tm/sz:7.3f} ns/kernel -- {sz*FLOPS/tm:10.3f} GFLOPS")

