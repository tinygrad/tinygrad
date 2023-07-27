#!/usr/bin/env python

import numpy as np
import pyopencl as cl

a_np = np.array([0,1,2,3]).astype(np.float16)
print(a_np)
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)

SRC = r"""//CL//
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void E_2_4(__global unsigned char* data0, const __global half* data1) {
        int gidx0 = get_group_id(0);  /* 2 */

        float4 val1_0 = vload_half4(0, data1+gidx0*4);
        data0[(gidx0 * 4)] = val1_0.x;
        data0[(gidx0 * 4) + 1] = val1_0.y;
        data0[(gidx0 * 4) + 2] = val1_0.z;
        data0[(gidx0 * 4) + 3] = val1_0.w;
    }
"""

prg = cl.Program(ctx, SRC).build()
b_np = np.array([0,0,0,0]).astype(np.uint8)
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, b_np.nbytes)
prg.E_2_4(queue, a_np.shape, None, res_g, a_g)
print(str(prg.get_info(cl.program_info.BINARIES)[0]))
res_np = np.empty_like(b_np)
cl.enqueue_copy(queue, res_np, res_g)

# Check on CPU with Numpy:
print(res_np) # prints [0 0 2 3]