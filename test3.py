from tinygrad.runtime.ops_gpu import CL
from tinygrad.runtime.ops_shm import RawShmBuffer
from tinygrad.tensor import Tensor
import multiprocessing.shared_memory as shared_memory
import numpy as np
import time

# for device in ["gpu:0"]:
#   for N in [2, 4, 8, 16, 32]:
#     t = Tensor.rand(2048, 2048, N, device=device).realize()
#     shm_name = (s := shared_memory.SharedMemory(create=True, size=t.nbytes())).name
#     rb = RawShmBuffer(t.numel(), t.dtype, device=f"{shm_name},{device}:{N}")
#
#     times = []
#     for _ in range(20):
#       st = time.perf_counter()
#       t.lazydata.realized._copyout(np.frombuffer(rb._buffer(), dtype=t.dtype.np))
#       times.append(time.perf_counter() - st)
#     best_time = min(times)
#     print(f"{device}, {best_time:.5f} s, {t.nbytes() / best_time / 1e9:.5f} GB/s, {2048}x{2048}x{N}, {t.nbytes() / 1e3:.5f} KB")

for N in [2, 4, 8, 16, 32]:
  t = Tensor.rand(2048, 2048, N, device="gpu:0").realize()

  times = []
  for _ in range(20):
    st = time.perf_counter()
    t2 = t.to("gpu:1").realize()
    times.append(time.perf_counter() - st)
  best_time = min(times)
  print(f"{best_time:.5f} s, {t.nbytes() / best_time / 1e9:.5f} GB/s, {2048}x{2048}x{N}, {t.nbytes() / 1e3:.5f} KB")
