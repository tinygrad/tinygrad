import numpy as np
from tinygrad.runtime.opencl import CLBuffer, CLProgram, CLImage, OSX_TIMING_RATIO

# M1 Max
# 32 core GPU
# Each GPU core is split into 16 Execution Units, which each contain eight Arithmetic Logic Units (ALUs)
# In total, the M1 Max GPU contains up to 512 execution units or 4096 ALUs.
# M1 Max delivers up to 400GB/s of memory bandwidth

# returns ns
def benchmark(prog):
  e = prog()
  e.wait()
  return ((e.profile.end - e.profile.start) * OSX_TIMING_RATIO)
def mb(prog, N=10): return min([benchmark(prog) for _ in range(N)])

MAX = 23
buffer_sz = 2**(MAX-1)*16
print(f"buffers using {3*2**MAX*16*1e-6:.2f} MB")
a = CLBuffer(buffer_sz)
b = CLBuffer(buffer_sz)
c = CLImage((1024, 1024, 4))
#rd = np.empty(shape=(buffer_sz//4,), dtype=np.float32)

print("*** empty kernel launch ***")
prog = CLProgram("test", "__kernel void test(__global float4 *a, __global float4 *b) { }")
for sz in [2**i for i in range(10,MAX)][::-1]:
  tm = mb(lambda: prog([sz,1,1], None, a._cl, b._cl))
  print(f"{sz:10d} {tm*1e-3:9.2f} us {tm/sz:7.3f} ns/kernel")

print("*** speed of global memory (L2) ***")
prog = CLProgram("test", """__kernel void test(__global float4 *a, __global float4 *b) {
  int gid = get_global_id(0);
  a[gid] = b[gid];
}""")
for sz in [2**i for i in range(10,MAX)][::-1]:
  tm = mb(lambda: prog([sz,1,1], None, a._cl, b._cl))
  print(f"{sz:10d} {tm*1e-3:9.2f} us {tm/sz:7.3f} ns/kernel -- {(sz*16*2)/tm:8.3f} GB/s")

print("*** speed of global memory (L1 cached) ***")
prog = CLProgram("test", """__kernel void test(__global float4 *a, __global float4 *b) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  float4 acc = 0;
  for (int i = lid; i < 256+lid; i++) { acc += b[i]; }
  a[gid] = acc;
}""")
for sz in [2**i for i in range(10,MAX)][::-1]:
  tm = mb(lambda: prog([sz,1,1], [256,1,1], a._cl, b._cl))
  print(f"{sz:10d} {tm*1e-3:9.2f} us {tm/sz:7.3f} ns/kernel -- {(sz*16*256)/tm:10.3f} GB/s L1 cache")

print("*** speed of texture memory (L1 cached) ***")
prog = CLProgram("test", """__kernel void test(__global float4 *a, read_only image2d_t c) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  float4 acc = 0;
  for (int i = lid; i < 256+lid; i++) { acc += read_imagef(c, smp, (int2)(0,i)); }
  a[gid] = acc;
}""")
for sz in [2**i for i in range(10,MAX)][::-1]:
  tm = mb(lambda: prog([sz,1,1], [256,1,1], a._cl, c._cl))
  print(f"{sz:10d} {tm*1e-3:9.2f} us {tm/sz:7.3f} ns/kernel -- {(sz*16*256)/tm:10.3f} GB/s L1 texture cache")

print("*** speed of local memory ***")
prog = CLProgram("test", """__kernel void test(__global float4 *a, __global float4 *b) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  __local float4 lmem[512];
  lmem[lid] = (float4)(lid, lid, lid, lid);
  lmem[lid+256] = (float4)(lid, lid, lid, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  float4 acc = 0;
  for (int i = lid; i < 256+lid; i++) { acc += lmem[i]; }
  a[gid] = acc;
}""")
for sz in [2**i for i in range(10,MAX)][::-1]:
  tm = mb(lambda: prog([sz,1,1], [256,1,1], a._cl, b._cl))
  print(f"{sz:10d} {tm*1e-3:9.2f} us {tm/sz:7.3f} ns/kernel -- {(sz*16*256)/tm:10.3f} GB/s local memory read")


print("*** speed of FMAs ***")
prog = CLProgram("test", """__kernel void test(__global float4 *a, __global float4 *b) {
  int gid = get_global_id(0);
  float4 r0 = (float4)(0.0, 8.0, 4.0, 4.0);
  float4 r1 = (float4)(1.0, 9.0, 4.0, 8.0);
  float4 r2 = (float4)(2.0, 10.0, 4.0, 4.0);
  float4 r3 = (float4)(3.0, 11.0, 3.0, 9.0);
  float4 b0 = (float4)(4.0, 12.0, 4.0, 4.0);
  float4 b1 = (float4)(5.0, 13.0, 4.0, 2.0);
  float4 b2 = (float4)(6.0, 14.0, 2.0, 4.0);
  float4 b3 = (float4)(7.0, 15.0, 4.0, 3.0);
  float4 acc = 0;
  for (int i = 0; i < 256; i++) {
    acc += r0 * b0;
    acc += r0 * b1;
    acc += r0 * b2;
    acc += r0 * b3;
    acc += r1 * b0;
    acc += r1 * b1;
    acc += r1 * b2;
    acc += r1 * b3;
    acc += r2 * b0;
    acc += r2 * b1;
    acc += r2 * b2;
    acc += r2 * b3;
    acc += r3 * b0;
    acc += r3 * b1;
    acc += r3 * b2;
    acc += r3 * b3;
  }
  a[gid] = acc;
}""")
for sz in [2**i for i in range(10,MAX)][::-1]:
  #      fma  float4   statements    loop
  FLOPS = 2 *   4    *   16       *   256
  tm = mb(lambda: prog([sz,1,1], None, a._cl, b._cl))
  print(f"{sz:10d} {tm*1e-3:9.2f} us {tm/sz:7.3f} ns/kernel -- {sz*FLOPS/tm:10.3f} GFLOPS")

