import time
import numpy as np
from tinygrad.helpers import dtypes, getenv
from tinygrad.runtime.ops_hip import RawHIPBuffer, HIPProgram

N = getenv("N", 1024)
assert N%16 == 0, "multiple of 16"
FLOPS = N*N*N*2
BW = N*N*3*4

a = RawHIPBuffer(N*N, dtypes.float32)

nb = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32).astype(np.float16)
nc = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32).astype(np.float16)
b = RawHIPBuffer.fromCPU(nb)
c = RawHIPBuffer.fromCPU(nc)

prog = HIPProgram("test", f"""
typedef float float8 __attribute__((ext_vector_type(8)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
extern "C" __global__ void test(float* c, __half* a, __half* b) {{
  const int gx = blockIdx.x;
  const int gy = blockIdx.y;

  c += gx*16*{N} + gy*16;
  a += gx*16*{N};
  b += gy*16;

  const int lIdx = threadIdx.x;
  const int lane = lIdx % 16;

  half16 a_frag;
  half16 b_frag;
  float8 c_frag = {{}};

  for (int k = 0; k < {N}; k += 16) {{
    for (int ele = 0; ele < 16; ++ele) {{
      a_frag[ele] = a[{N}*lane + (k+ele)];
      b_frag[ele] = b[(k+ele)*{N} + lane];
    }}

    c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);
  }}

  for (int ele = 0; ele < 8; ++ele) {{
    const int r = ele * 2 + (lIdx / 16);
    c[{N}*r + lane] = c_frag[ele];
  }}
}}""")

def timeit(fxn):
  st = time.perf_counter()
  et = fxn()
  ret = time.perf_counter() - st # NOTE: et doesn't contain the launch overhead
  #print(f"{ret*1e6:.2f} us")
  return ret

tm = min([timeit(lambda: prog([N//16, N//16, 1], [32, 1, 1], a, b, c, wait=True)) for _ in range(20)])
na = a.toCPU().reshape(N,N)
comp = nb.astype(np.float32) @ nc.astype(np.float32)
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul, {BW*1e-9/tm:.2f} GB/s")
np.testing.assert_allclose(na, comp, atol=1e-2, rtol=1e-2)