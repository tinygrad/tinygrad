import numpy as np
from tinygrad.runtime.opencl import CLBuffer, CLProgram, OSX_TIMING_RATIO

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

MAX = 24
buffer_sz = 2**(MAX-1)*16
print(f"buffers using {3*2**MAX*16*1e-6:.2f} MB")
a = CLBuffer(buffer_sz)
b = CLBuffer(buffer_sz)
#rd = np.empty(shape=(buffer_sz//4,), dtype=np.float32)

print("*** empty kernel launch ***")
prog = CLProgram("empty", "__kernel void empty(__global float4 *a, __global float4 *b) { }")
for sz in [2**i for i in range(10,MAX)][::-1]:
  tm = mb(lambda: prog([sz,1,1], None, a._cl, b._cl))
  print(f"{sz:10d} {tm*1e-3:9.2f} us {tm/sz:7.3f} ns/kernel")

print("*** speed of global memory (L2) ***")
prog = CLProgram("copy", """__kernel void copy(__global float4 *a, __global float4 *b) {
  int gid = get_global_id(0);
  a[gid] = b[gid];
}""")
for sz in [2**i for i in range(10,MAX)][::-1]:
  tm = mb(lambda: prog([sz,1,1], None, a._cl, b._cl))
  print(f"{sz:10d} {tm*1e-3:9.2f} us {tm/sz:7.3f} ns/kernel -- {(sz*16*2)/tm:7.3f} GB/s")

print("*** speed of local memory ***")
prog = CLProgram("copy", """__kernel void copy(__global float4 *a, __global float4 *b) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  __local float4 lmem[256];
  lmem[lid] = (float4)(4.0, 4.0, 4.0, 4.0);
  barrier(CLK_LOCAL_MEM_FENCE);
  float4 acc = 0;
  //for (int i = lid; i < 256; i++) { acc += lmem[i]; }
  //for (int i = 0; i < lid; i++) { acc += lmem[i]; }
  for (int i = 0; i < 256; i++) { acc += lmem[i]; }
  a[gid] = acc;
}""")
for sz in [2**i for i in range(10,MAX)][::-1]:
  tm = mb(lambda: prog([sz,1,1], [256,1,1], a._cl, b._cl))
  print(f"{sz:10d} {tm*1e-3:9.2f} us {tm/sz:7.3f} ns/kernel -- {(sz*16*256)/tm:10.3f} GB/s local memory read")


# speed of FMAs

