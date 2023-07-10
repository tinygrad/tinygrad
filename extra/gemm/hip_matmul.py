import time
import numpy as np
from tinygrad.helpers import dtypes, getenv
from tinygrad.runtime.ops_hip import RawHIPBuffer, HIPProgram

# AMD_LOG_LEVEL=3 ./MIOpenDriver gemm --iter 1000 --time 1 --a_w 2048 --a_h 2048 --b_w 2048
# Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS3_ASE_ASGT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_DTL0_DTVA0_DVO0_ETSP_EPS1_FL0_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPP128_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS1_SU32_SUM0_SUS128_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT4_64_TLDS1_USFGROn1_VAW2_VSn1_VW4_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM4
# gets ~100
# hipExtModuleLaunchKernel ( 0x0x16ccde0, 2048, 16, 1, 128, 1, 1,

# we only get ~34
# KY=2 KX=2 N=2048 python3 extra/gemm/hip_matmul.py
#   4194304    502.57 us, would be  34184.30 GFLOPS matmul, 100.15 GB/s

N = getenv("N", 64)
KX = getenv("KX", 1)
KY = getenv("KY", 1)
assert N%(16*KX) == 0, f"N must be multiple of {16*KX}"
assert N%(16*KY) == 0, f"N must be multiple of {16*KY}"
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
  const int gy = blockIdx.y*4 + threadIdx.y;

  c += gx*{KX*16}*{N} + gy*{KY*16};
  a += gx*{KX*16}*{N};
  b += gy*{KY*16};

  const int lIdx = threadIdx.x;
  const int lane = lIdx%16;

  half16 a_frag[{KX}];
  half16 b_frag[{KY}];
  float8 c_frag[{KY}][{KX}] = {{}};

  for (int k = 0; k < {N}; k += 16) {{
    for (int ele = 0; ele < 16; ++ele) {{
      for (int x = 0; x < {KX}; x++) {{
        a_frag[x][ele] = a[{N}*lane + (k+ele) + x*{16*N}];
      }}
    /*}}
    for (int ele = 0; ele < 16; ++ele) {{*/
      for (int y = 0; y < {KY}; y++) {{
        b_frag[y][ele] = b[(k+ele)*{N} + lane + y*16];
      }}
    }}
    for (int y = 0; y < {KY}; y++) {{
      for (int x = 0; x < {KX}; x++) {{
        c_frag[y][x] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag[x], b_frag[y], c_frag[y][x]);
      }}
    }}
  }}

  for (int ele = 0; ele < 8; ++ele) {{
    const int r = ele * 2 + (lIdx / 16);
    for (int y = 0; y < {KY}; y++) {{
      for (int x = 0; x < {KX}; x++) {{
        c[{N}*r + lane + y*16 + x*{16*N}] = c_frag[y][x][ele];
      }}
    }}
  }}
}}""")

def timeit(fxn):
  st = time.perf_counter()
  et = fxn()
  ret = time.perf_counter() - st # NOTE: et doesn't contain the launch overhead
  #print(f"{ret*1e6:.2f} us")
  return et

# 2048, 16, 1, 128, 1, 1
tm = min([timeit(lambda: prog([N//(KX*16), N//(KY*16*4), 1], [32, 4, 1], a, b, c, wait=True)) for _ in range(20)])
na = a.toCPU().reshape(N,N)
comp = nb.astype(np.float32) @ nc.astype(np.float32)
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul, {BW*1e-9/tm:.2f} GB/s")
np.testing.assert_allclose(na, comp, atol=1e-2, rtol=1e-2)