from tinygrad.runtime.ops_gpu import CLBuffer, CLProgram
from tinygrad.helpers import dtypes
from extra.helpers import Timing

if __name__ == "__main__":
  SZ = 1024*1024*256
  b0 = CLBuffer(SZ, dtypes.float32, device=0)
  b1 = CLBuffer(SZ, dtypes.float32, device=1)
  for i in range(5):
    print("copies")
    with Timing(on_exit=lambda x: f" {(SZ*b0.dtype.itemsize)/x:.2f} GB/s"):
      c0 = b0.toCPU()
    with Timing(on_exit=lambda x: f" {(SZ*b1.dtype.itemsize)/x:.2f} GB/s"):
      c1 = b1.toCPU()


