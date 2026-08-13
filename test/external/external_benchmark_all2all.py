from tinygrad import Context, Device, Tensor, TinyJit, dtypes
from tinygrad.helpers import Timing

GPUS, DEPTH, WARMUP = 8, 4, 3
devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(GPUS))

with Context(ALL2ALL=1, JIT_BATCH_SIZE=0):
  for size in (4194304, 6291456, 14680064, 29360128, 131334144, 134217728):
    bufs = tuple(Tensor.empty(size, dtype=dtypes.uint8, device=dev).contiguous().realize() for _ in range(DEPTH) for dev in devs)
    @TinyJit
    def all_to_all(*bufs:Tensor):
      return Tensor.realize(*(src.to(dst) for i,src in enumerate(bufs) for j,dst in enumerate(devs) if i % GPUS != j))
    for i in range(-WARMUP, 5):
      with Timing(f"{size/2**20:6.1f} MiB  {i+1}/5  ", lambda ns: f"  {size*GPUS*(GPUS-1)*DEPTH/ns:8.2f} GB/s", enabled=i>=0):
        all_to_all(*bufs)
        for dev in devs: Device[dev].synchronize()
