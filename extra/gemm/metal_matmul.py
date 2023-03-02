import numpy as np
from tinygrad.runtime.ops_metal import CLBuffer, CLProgram

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
#include <metal_stdlib>
#include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+
using namespace metal;
kernel void test(device float *a, device const float *data1, device const float *data2, uint3 gid [[thread_position_in_grid]], uint3 xid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]], uint sidx [[simdgroup_index_in_threadgroup]]) {{
  // 1-2 simd groups
  //uint idx = gid.x/32;
  //uint pos_x = (idx%{N//32}) * 32;
  //uint pos_y = (idx/{N//32}) * 32;

  // 4 simd groups
  uint idx = gid.x/128;
  uint pos_x = (idx%{N//64}) * 64;
  uint pos_y = (idx/{N//64}) * 64;
  pos_x += (sidx%2) * 32;
  pos_y += (sidx/2) * 32;

  // 16 simd groups (slow)
  /*uint idx = gid.x/512;
  uint pos_x = (idx%{N//128}) * 128;
  uint pos_y = (idx/{N//128}) * 128;
  pos_x += (sidx%4) * 32;
  pos_y += (sidx/4) * 32;*/

  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; i++) {{
    for (uint j = 0; j < 4; j++) {{
      acc[i][j] = simdgroup_float8x8(0);
    }}
  }}
  simdgroup_float8x8 A[4];
  simdgroup_float8x8 B[4];
  data1 += pos_x * {N};
  data2 += pos_y;

  for (uint k = 0; k < {N}; k+=8) {{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_load(A[0], data1, {N}, ulong2(k, 0));
    simdgroup_load(A[1], data1, {N}, ulong2(k, 8));
    simdgroup_load(A[2], data1, {N}, ulong2(k, 16));
    simdgroup_load(A[3], data1, {N}, ulong2(k, 24));
    simdgroup_load(B[0], data2, {N}, ulong2(0, k));
    simdgroup_load(B[1], data2, {N}, ulong2(8, k));
    simdgroup_load(B[2], data2, {N}, ulong2(16, k));
    simdgroup_load(B[3], data2, {N}, ulong2(24, k));

    simdgroup_multiply_accumulate(acc[0][0], A[0], B[0], acc[0][0]);
    simdgroup_multiply_accumulate(acc[0][1], A[1], B[0], acc[0][1]);
    simdgroup_multiply_accumulate(acc[0][2], A[2], B[0], acc[0][2]);
    simdgroup_multiply_accumulate(acc[0][3], A[3], B[0], acc[0][3]);
    simdgroup_multiply_accumulate(acc[1][0], A[0], B[1], acc[1][0]);
    simdgroup_multiply_accumulate(acc[1][1], A[1], B[1], acc[1][1]);
    simdgroup_multiply_accumulate(acc[1][2], A[2], B[1], acc[1][2]);
    simdgroup_multiply_accumulate(acc[1][3], A[3], B[1], acc[1][3]);
    simdgroup_multiply_accumulate(acc[2][0], A[0], B[2], acc[2][0]);
    simdgroup_multiply_accumulate(acc[2][1], A[1], B[2], acc[2][1]);
    simdgroup_multiply_accumulate(acc[2][2], A[2], B[2], acc[2][2]);
    simdgroup_multiply_accumulate(acc[2][3], A[3], B[2], acc[2][3]);
    simdgroup_multiply_accumulate(acc[3][0], A[0], B[3], acc[3][0]);
    simdgroup_multiply_accumulate(acc[3][1], A[1], B[3], acc[3][1]);
    simdgroup_multiply_accumulate(acc[3][2], A[2], B[3], acc[3][2]);
    simdgroup_multiply_accumulate(acc[3][3], A[3], B[3], acc[3][3]);
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

