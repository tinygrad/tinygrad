# tiny@tiny9:~/tinygrad$ python3 examples/benchmark_copies.py
# CPU copy 6.18 ms, 16.28 GB/s
# GPU copy 4.38 ms, 23.00 GB/s
# GPU  6x  1.85 ms, 54.54 GB/s

from tinygrad.helpers import Timing
import time
def timeit(fxn):
  tms = []
  for _ in range(10):
    st = time.perf_counter()
    fxn()
    tms.append(time.perf_counter() - st)
  return min(tms)

import ctypes
N = 16384
sz_bytes = N * N * 6
print(f"buffer size {sz_bytes/1e6:.2f} MB")
import extra.hip_wrapper as hip
inp = hip.hipHostMalloc(sz_bytes)
out = hip.hipHostMalloc(sz_bytes)

# ***** CPU timing *****

def cpu_memcpy(): ctypes.memmove(out, inp, sz_bytes)
print(f"CPU copy {(tm:=timeit(cpu_memcpy))*1000:6.2f} ms, {sz_bytes*1e-9/tm:.2f} GB/s")

# ***** multiCPU timing *****

import threading
THREADS = 16
sz_bytes_chunk = sz_bytes//THREADS
def multicpu_memcpy():
  ts = [threading.Thread(target=ctypes.memmove, args=(out+sz_bytes_chunk*i, inp+sz_bytes_chunk*i, sz_bytes_chunk)) for i in range(THREADS)]
  for t in ts: t.start()
  for t in ts: t.join()
print(f"CPU  {THREADS:2d}x {(tm:=timeit(multicpu_memcpy))*1000:6.2f} ms, {sz_bytes*1e-9/tm:.2f} GB/s")

# ***** GPU timing *****

STREAMS = 16
sz_bytes_chunk = sz_bytes//STREAMS
buf = [hip.hipMalloc(sz_bytes_chunk) for _ in range(STREAMS)]
streams = [hip.hipStreamCreate() for _ in range(STREAMS)]
def gpu_roundtrip():
  for i in range(STREAMS):
    hip.hipMemcpyAsync(buf[i], ctypes.c_void_p(inp+sz_bytes_chunk*i), sz_bytes_chunk, hip.hipMemcpyHostToDevice, streams[i])
    hip.hipMemcpyAsync(ctypes.c_void_p(out+sz_bytes_chunk*i), buf[i], sz_bytes_chunk, hip.hipMemcpyDeviceToHost, streams[i])
  hip.hipDeviceSynchronize()
print(f"GPU copy {(tm:=timeit(gpu_roundtrip))*1000:6.2f} ms, {sz_bytes*1e-9/tm:.2f} GB/s")

# ***** multiGPU timing *****

STREAMS = 8
GPUS = 6
sz_bytes_chunk = sz_bytes//(STREAMS*GPUS)
buf = [hip.hipSetDevice(j) or [hip.hipMalloc(sz_bytes_chunk) for _ in range(STREAMS)] for j in range(GPUS)]
streams = [hip.hipSetDevice(j) or [hip.hipStreamCreate() for _ in range(STREAMS)] for j in range(GPUS)]
def multigpu_roundtrip():
  for i in range(STREAMS):
    for j in range(GPUS):
      hip.hipSetDevice(j)
      offset = sz_bytes_chunk * (j*STREAMS + i)
      hip.hipMemcpyAsync(buf[j][i], ctypes.c_void_p(inp+offset), sz_bytes_chunk, hip.hipMemcpyHostToDevice, streams[j][i])
      hip.hipMemcpyAsync(ctypes.c_void_p(out+offset), buf[j][i], sz_bytes_chunk, hip.hipMemcpyDeviceToHost, streams[j][i])
  for j in range(GPUS):
    hip.hipSetDevice(j)
    hip.hipDeviceSynchronize()
print(f"GPU   6x {(tm:=timeit(multigpu_roundtrip))*1000:6.2f} ms, {sz_bytes*1e-9/tm:.2f} GB/s")
