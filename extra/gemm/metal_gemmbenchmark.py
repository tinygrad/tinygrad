import time
from tinygrad.runtime.metal import CLBuffer, CLProgram

def benchmark(prog):
  st = time.monotonic_ns()
  e = prog()
  e.waitUntilCompleted()
  return time.monotonic_ns() - st
def mb(prog, N=10): return min([benchmark(prog) for _ in range(N)])

a = CLBuffer(2048*2048*4)
b = CLBuffer(2048*2048*4)
c = CLBuffer(2048*2048*4)

FLOPS = 2048*2048*2048*2

prog = CLProgram("test", """kernel void test(device float *a, device float *data1, device float *data2) {
  /*size_t idx1 = get_global_id(0);
  size_t idx0 = get_global_id(1);
  __local float ldata1[256];
  __local float ldata2[256];
  size_t lidx1 = get_local_id(0);
  size_t lidx0 = get_local_id(1);
  a[idx1*512+idx0] = acc;
  float4 acc = 0;
  for (int idx2 = 0; idx2 < 256; idx2++) {
    acc += idx2;
  }*/
  a[0] = 4;
  a[1] = 5;
}""")
tm = mb(lambda: prog([512,512], [8,8], a._cl, b._cl, c._cl))
print(f"{512*512:10d} {tm*1e-3:9.2f} us, would be {FLOPS/tm:.2f} GFLOPS matmul")
print(a.toCPU()[0:10])