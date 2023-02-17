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

MAX = 25
print(f"buffers using {2*2**MAX*16*1e-6} MB")
a = CLBuffer(2**MAX*16)
b = CLBuffer(2**MAX*16)

print("*** empty kernel launch ***")
prog = CLProgram("empty", "__kernel void empty() { }")
for sz in [2**i for i in range(10,MAX)][::-1]:
  tm = mb(lambda: prog([sz,1,1],None))
  print(f"{sz:10d} {tm*1e-3:9.2f} us {tm/sz:7.3f} ns/kernel")

print("*** speed of global memory (L2) ***")
prog = CLProgram("copy", """__kernel void copy(__global float4 *a, __global float4 *b) {
  int gid = get_global_id(0);
  a[gid] = b[gid];
}""")
for sz in [2**i for i in range(10,MAX)][::-1]:
  tm = mb(lambda: prog([sz,1,1],None, a._cl, b._cl))
  print(f"{sz:10d} {tm*1e-3:9.2f} us {tm/sz:7.3f} ns/kernel -- {(sz*16*2)/tm:7.3f} GB/s")

# speed of local memory
# speed of FMAs

