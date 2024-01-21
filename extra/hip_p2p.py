import time
from tinygrad.helpers import Timing, to_mv
from tinygrad.runtime.ops_hip import HIPDevice, compile_hip, check
from gpuctypes import hip

if __name__ == "__main__":
  d0 = HIPDevice("HIP")
  d1 = HIPDevice("HIP:1")

  lib = compile_hip('extern "C" __global__ void block(int *sem) { while (!atomicCAS_system(sem, 1, 1)); }')
  block = d0.runtime("block", lib)

  lib = compile_hip(f"""
    extern "C" __global__ void copy(float* a, float* b) {{
      const int gx = (blockIdx.x*blockDim.x + threadIdx.x)*4;
      a[gx] = b[gx];
      a[gx+1] = b[gx+1];
      a[gx+2] = b[gx+2];
      a[gx+3] = b[gx+3];
    }}""")
  prg = d0.runtime("copy", lib)

  lib = compile_hip('extern "C" __global__ void sync(int *sem, int value) { atomicExch_system(sem, value); }')
  prg_set = d1.runtime("sync", lib)

  sz = 4*1024*1024*1024
  #sz = 4*1024*1024
  b0 = d0.allocator.alloc(sz)
  b1 = d1.allocator.alloc(sz)
  sem = d0.allocator._hostalloc(4)
  host = to_mv(sem, 4)
  def clear():
    host[0] = 0
    host[1] = 0
    host[2] = 0
    host[3] = 0

  #sem = d1.allocator.alloc(4)
  print(b0, b1)

  #check(hip.hipSetDevice(0))
  #check(hip.hipDeviceEnablePeerAccess(1, 0))
  #check(hip.hipSetDevice(1))
  #check(hip.hipDeviceEnablePeerAccess(0, 0))

  #prg_set(sem, 32, global_size=[1,1,1], local_size=[1,1,1])
  #d1.synchronize()

  for i in range(3):
    with Timing("copy ", lambda x: f" {sz/x:.2f} GB/s"):
      clear()
      #prg_set(sem, 0, global_size=[1,1,1], local_size=[1,1,1])
      #d1.synchronize()
      block(sem, global_size=[1,1,1], local_size=[1,1,1])
      d0.allocator.transfer(b0, b1, sz)
      #prg(b0, b1, global_size=[((sz//4)//4)//32, 1, 1], local_size=[32, 1, 1])
      time.sleep(0.1)
      prg_set(sem, 1, global_size=[1,1,1], local_size=[1,1,1])
      #d1.synchronize()
      #print(to_mv(sem, 4)[0])
      d0.synchronize()


