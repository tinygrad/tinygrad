#!/usr/bin/env python3
import numpy as np
from tinygrad.runtime.ops_gpu import CLProgram, CLCompiler
from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from hexdump import hexdump

# https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md
# https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html
# https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html
# https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_split_matrix_multiply_accumulate.html
# https://hc34.hotchips.org/assets/program/conference/day1/GPU%20HPC/Intel_s%20Ponte%20Vecchio%20GPU%20-%20Architecture%20Systems%20and%20Software%20FINAL.pdf

device = Device["GPU"]

# NOTE: only the subgroup type 8 ones work
prog = CLProgram(device, "test", CLCompiler(device, "test").compile(f"""
__attribute__((reqd_sub_group_size(8)))
__kernel void test(__global float* data0, const __global int* data1, const __global int8* data2) {{
  int lidx0 = get_local_id(0);
  int a = data1[lidx0/2];
  int8 b = data2[lidx0];
  float out = intel_sub_group_f16_f16_matrix_mad_k16(a, b, 0.0f);
  data0[lidx0] = out;
}}
"""))
#with open("/tmp/test.elf", "wb") as f: f.write(prog.lib)

a = Buffer("GPU", 8, dtypes.float32)
b = Buffer("GPU", 0x10, dtypes.float16)
c = Buffer("GPU", 8*0x10, dtypes.float16)

row = np.array([0]*0x10, np.float16)
mat = np.zeros((8, 0x10), np.float16)
#row[0] = 1
#row[1] = 22
#row[2] = 3
#row[3] = 3
# 2-3 don't work
#row[4] = 3
#row[5] = 4
#row[7] = 4
# 6-7 don't work
#row[8] = 3

#row[3] = 4
mat[0][0] = 2
mat[0][1] = 3
mat[0][2] = 4
mat[0][4] = 9
mat[0][4] = 9
mat[0][5] = 8

mat[1][0] = 7
mat[1][1] = 9
mat[1][2] = 11
#mat[0][2] = 1
#mat += 1
#mat[0] += 1
#mat[1] += 1
#mat[0][0] = 2
#mat[0][1] = 3
#mat[1][0] = 8
#mat[1][1] = 9
#mat[0][2] = 6
#mat[0][2] = 6
#mat[1][0] = 4

row = np.array([1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8], np.float16)
mat = np.random.random((8, 0x10)).astype(np.float16)
mat[:, 8:] = 0

b.copyin(row.data)
c.copyin(mat.data)
ret = prog(a._buf, b._buf, c._buf, global_size=[1,1,1], local_size=[8,1,1], wait=True)
print(ret)
out = np.frombuffer(a.as_buffer(), np.float32)
real = row.astype(np.float32)@mat.T.astype(np.float32)
print("out:", out)
print("real", real)


#b.copyin(np.zeros(0x10, np.float16).data)
#c.copyin(np.zeros(0x10, np.float16).data)
#print(a.as_buffer().cast('f').tolist())
