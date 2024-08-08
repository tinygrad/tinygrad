import time
from tinygrad import Tensor, Device

if __name__ == "__main__":
  N = 8192
  A = Tensor.rand(N, N).shard(("NV:0", "NV:1"), 0).realize()
  B = Tensor.rand(N, N).shard(("NV:0", "NV:1"), 1).realize()
  print("***** MUL *****")
  for i in range(10):
    Device["NV:0"].synchronize()
    Device["NV:1"].synchronize()
    st = time.perf_counter()
    (A@B).realize()
    Device["NV:0"].synchronize()
    Device["NV:1"].synchronize()
    et = time.perf_counter()
    print(f"{(N*N*N*2*1e-12)/(et-st):.2f} TFLOPS")
