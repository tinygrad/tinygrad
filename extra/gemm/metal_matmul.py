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
#pragma METAL internals : enable
using namespace metal;
kernel void test(device float *a, device float *data1, device float *data2, uint3 gid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
  int idx = gid.x/32;
  int pos_x = (idx%{N//16}) * 16;
  int pos_y = (idx/{N//16}) * 16;
  simdgroup_float8x8 acc[2][2];
  acc[0][0] = simdgroup_float8x8(0);
  acc[0][1] = simdgroup_float8x8(0);
  acc[1][0] = simdgroup_float8x8(0);
  acc[1][1] = simdgroup_float8x8(0);
  simdgroup_float8x8 A[2];
  simdgroup_float8x8 B[2];
  //__metal_get_null_simdgroup_event
  //__metal_simdgroup_async_copy_2d
  for (uint k = 0; k < {N}; k+=8) {{
    simdgroup_load(A[0], data1, {N}, ulong2(k, pos_x));
    simdgroup_load(A[1], data1, {N}, ulong2(k, pos_x+8));
    simdgroup_load(B[0], data2, {N}, ulong2(pos_y, k));
    simdgroup_load(B[1], data2, {N}, ulong2(pos_y+8, k));
    simdgroup_multiply_accumulate(acc[0][0], A[0], B[0], acc[0][0]);
    simdgroup_multiply_accumulate(acc[0][1], A[1], B[0], acc[0][1]);
    simdgroup_multiply_accumulate(acc[1][0], A[0], B[1], acc[1][0]);
    simdgroup_multiply_accumulate(acc[1][1], A[1], B[1], acc[1][1]);
  }}
  simdgroup_store(acc[0][0], a, {N}, ulong2(pos_y, pos_x));
  simdgroup_store(acc[0][1], a, {N}, ulong2(pos_y, pos_x+8));
  simdgroup_store(acc[1][0], a, {N}, ulong2(pos_y+8, pos_x));
  simdgroup_store(acc[1][1], a, {N}, ulong2(pos_y+8, pos_x+8));
}}""")
tm = mb(lambda: prog([N*N//8], [32], a._cl, b._cl, c._cl))
na = a.toCPU().reshape(N,N)
comp = nb@nc
if N <= 32:
  print(na)
  print(comp)
print(f"{N*N:10d} {tm*1e-3:9.2f} us, would be {FLOPS/tm:.2f} GFLOPS matmul")
np.testing.assert_allclose(na, comp, atol=1e-3)

