# copying the kernels from https://github.com/microsoft/ArchProbe into Python
import numpy as np
from tinygrad.runtime.ops_gpu import CLProgram, CLBuffer
from tinygrad.helpers import dtypes
from tqdm import trange, tqdm
from matplotlib import pyplot as plt

def reg_count(nthread, ngrp, nreg):
  reg_declr = ''.join([f"float reg_data{i} = (float)niter + {i};\n" for i in range(nreg)])
  reg_comp = ''.join([f"reg_data{i} *= {(i-1)%nreg};\n" for i in range(nreg)])
  reg_reduce = ''.join([f"out_buf[{i}] = reg_data{i};\n" for i in range(nreg)])
  prg = f"""__kernel void reg_count(
    __global float* out_buf,
    __private const int niter
  ) {{
    {reg_declr}
    int i = 0;
    for (; i < niter; ++i) {{
      {reg_comp}
    }}
    i = i >> 31;
    {reg_reduce}
  }}"""
  out_buf = CLBuffer(1, dtypes.float32)
  cl = CLProgram("reg_count", prg, argdtypes=[None, np.int32])
  return min([cl([nthread, ngrp, 1], [nthread, 1, 1], out_buf, 10, wait=True) for _ in range(10)])

"""
print("probing registers")
pts = [(nreg, reg_count(1, 1, nreg)) for nreg in trange(1, 257)]   # archprobe goes to 512
plt.plot(*zip(*pts))
plt.show()
"""

def buf_cache_hierarchy_pchase(ndata, stride=1):
  NCOMP = 16 # 64 byte is under the 128 byte cache line
  print("probe", ndata*NCOMP*4)
  prg = """__kernel void buf_cache_hierarchy_pchase(
    __global int16* src,
    __global int* dst,
    const int niter
  ) {
    int idx = 0;
    for (int i = 0; i < niter; ++i) {
      idx = src[idx].x;
    }
    *dst = idx;
  }"""
  idx_buf = np.zeros(ndata*NCOMP, dtype=np.int32)
  for i in range(ndata):
    idx_buf[i*NCOMP] = (i + stride) % ndata
  in_buf = CLBuffer.fromCPU(idx_buf)
  out_buf = CLBuffer(1, dtypes.int32)
  cl = CLProgram("buf_cache_hierarchy_pchase", prg, argdtypes=[None, None, np.int32])
  return min([cl([1, 1, 1], [1, 1, 1], in_buf, out_buf, ndata*4, wait=True) for _ in range(5)])

# 768 kb is real
print("probing cache size")
base = buf_cache_hierarchy_pchase(1, 191)
szs = list(range(128, 1024, 128)) + list(range(1024, 16*1024, 1024)) + list(range(16*1024, int(1.5*1024*1024), 16*1024)) #+ list(range(2*1024*1024, 20*1024*1024, 1024*1024))
pts = [(ndata, (buf_cache_hierarchy_pchase(ndata//64, 136329190282766681843115968953)-base)/ndata) for ndata in tqdm(szs)]
plt.plot(*zip(*pts))
plt.show()
