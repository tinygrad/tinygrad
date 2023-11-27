from tinygrad.runtime.ops_opencl import CLDevice

if __name__ == "__main__":
  dev = CLDevice()
  a, b, c = dev.buffer(10), dev.buffer(10), dev.buffer(10)
  prg: bytes = CLDevice.compiler("__kernel void add(__global float *a, __global float *b, __global float *c) { int i = get_global_id(0); c[i] = a[i] + b[i]; }")

  #from hexdump import hexdump
  #hexdump(prg)

  runtime = dev.runtime("add", prg)
  #runtime(a, b, c, global_size=(10,))
