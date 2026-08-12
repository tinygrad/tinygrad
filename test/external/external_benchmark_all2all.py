import time
from tinygrad import Context, Device, Tensor, TinyJit, dtypes
from tinygrad.helpers import colored

GPUS = 8
ITERS = 5
WARMUP = 3
DEPTH = 4
SIZES = (4194304, 6291456, 14680064, 29360128, 131334144, 134217728)

devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(GPUS))

for size in SIZES:
  sources = tuple(Tensor.empty(size, dtype=dtypes.uint8, device=dev).contiguous().realize() for _ in range(DEPTH) for dev in devices)

  @TinyJit
  def all_to_all(*srcs:Tensor):
    return Tensor.realize(*(src.to(dst) for i,src in enumerate(srcs) for j,dst in enumerate(devices) if i % GPUS != j))

  size_mib = size / 2**20
  times = []
  with Context(ALL2ALL=1, JIT_BATCH_SIZE=0):
    for iteration in range(WARMUP + ITERS):
      st = time.perf_counter()
      outputs = all_to_all(*sources)
      for dev in devices: Device[dev].synchronize()
      elapsed = time.perf_counter() - st
      if iteration >= WARMUP:
        times.append(elapsed)
        bw = size*GPUS*(GPUS-1)*DEPTH/elapsed/1e9
        print(f"{size_mib:6.1f} MiB  {iteration-WARMUP+1:2d}/{ITERS:<2d}  {elapsed*1e3:8.3f} ms  {bw:8.2f} GB/s")

  best = min(times)
  bw = size*GPUS*(GPUS-1)*DEPTH/best/1e9
  print(f"{size_mib:6.1f} MiB  {iteration-WARMUP+1:2d}/{ITERS:<2d}  {elapsed*1e3:8.3f} ms  {colored(f'{bw:8.2f} GB/s', 'green')}")
