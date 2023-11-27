import numpy as np
from tinygrad.runtime.ops_cl import CLDevice

if __name__ == "__main__":
  dev = CLDevice()
  a, b, c = dev.alloc(12), dev.alloc(12), dev.alloc(12)
  prg: bytes = CLDevice.compiler("__kernel void add(__global float *a, __global float *b, __global float *c) { int i = get_global_id(0); c[i] = a[i] + b[i]; }")
  runtime = dev.runtime("add", prg)

  a_cpu = np.array([1,2,3], dtype=np.float32)
  b_cpu = np.array([3,2,1], dtype=np.float32)
  c_cpu = np.array([-1,-1,-1], dtype=np.float32)

  dev.copyin(a, a_cpu.data)
  dev.copyin(b, b_cpu.data)
  tm = runtime(a, b, c, global_size=(3,), wait=True)
  print(tm)
  dev.copyout(c_cpu.data, c)

  print(c_cpu)
