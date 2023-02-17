import numpy as np
from tinygrad.runtime.opencl import CL, CLProgram, CLBuffer

out = np.zeros((1,), dtype=np.float32)

#LS = [1024,1,1]
LS = None
a = CLBuffer(4)

prog = CLProgram("test", "__kernel void test(__global float *a) { a[0] = 0; }")
prog([2048*2048//4,1,1], LS, a._cl)
prog([2048*2048//4,1,1], LS, a._cl)
prog([2048*2048//4,1,1], LS, a._cl)
CL.cl_queue.finish()

print("**** SETUP DONE ****")
prog([2048*2048//4,1,1], LS, a._cl)
CL.cl_queue.finish()
print("**** RUN DONE ****")

a.copyout(out)
print(out)
