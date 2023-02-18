import numpy as np
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

nb = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32) #.astype(np.int32).astype(np.float32)
nc = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32) #.astype(np.int32).astype(np.float32)
#nb = np.eye(N)
#nc = np.eye(N)
#nb = np.ones((N,N))
#nc = np.ones((N,N))
b.copyin(nb)
c.copyin(nc)

FLOPS = N*N*N*2

prog = CLProgram("test", f"""
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void test(device float *a, device float *data1, device float *data2, uint3 gid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
  //int simd = lid.x/32;
  int idx = gid.x/32;
  int pos_x = (idx%{N//32}) * 32;
  int pos_y = (idx/{N//32}) * 32;
  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; i++) {{
    for (uint j = 0; j < 4; j++) {{
      acc[i][j] = simdgroup_float8x8(0);
    }}
  }}
  simdgroup_float8x8 A[4];
  simdgroup_float8x8 B[4];
  //__metal_get_null_simdgroup_event
  //__metal_simdgroup_async_copy_2d
  for (uint k = 0; k < {N}; k+=8) {{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_load(A[0], data1, {N}, ulong2(k, pos_x));
    simdgroup_load(A[1], data1, {N}, ulong2(k, pos_x+8));
    simdgroup_load(A[2], data1, {N}, ulong2(k, pos_x+16));
    simdgroup_load(A[3], data1, {N}, ulong2(k, pos_x+24));
    simdgroup_load(B[0], data2, {N}, ulong2(pos_y, k));
    simdgroup_load(B[1], data2, {N}, ulong2(pos_y+8, k));
    simdgroup_load(B[2], data2, {N}, ulong2(pos_y+16, k));
    simdgroup_load(B[3], data2, {N}, ulong2(pos_y+24, k));

    for (uint i = 0; i < 4; i++) {{
      // must be manually unrolled
      simdgroup_multiply_accumulate(acc[i][0], A[0], B[i], acc[i][0]);
      simdgroup_multiply_accumulate(acc[i][1], A[1], B[i], acc[i][1]);
      simdgroup_multiply_accumulate(acc[i][2], A[2], B[i], acc[i][2]);
      simdgroup_multiply_accumulate(acc[i][3], A[3], B[i], acc[i][3]);
    }}
  }}
  for (uint i = 0; i < 4; i++) {{
    for (uint j = 0; j < 4; j++) {{
      simdgroup_store(acc[i][j], a, {N}, ulong2(pos_y+i*8, pos_x+j*8));
    }}
  }}
}}""")
tm = mb(lambda: prog([N*N//(2*4*4)], [4*32], a._cl, b._cl, c._cl))
na = a.toCPU().reshape(N,N)
comp = nb@nc
if N <= 32:
  print(na)
  print(comp)
print(f"{N*N:10d} {tm*1e-3:9.2f} us, would be {FLOPS/tm:.2f} GFLOPS matmul")
np.testing.assert_allclose(na, comp, atol=1e-3)

