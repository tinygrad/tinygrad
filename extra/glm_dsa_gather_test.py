# compare fp32 and paired-fp16 gathers on values already representable in fp16; time repeated calls
import time
from tinygrad import Tensor, Device, TinyJit, getenv
from tinygrad.llm.kernels.amd_glm_dsa import host_allgather, host_allgather_half
from tinygrad.llm.glm_dsa import _open_devices
N = getenv('SHARD', 14)
devs = tuple(Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(N))
_open_devices(devs)
n = getenv('N', 11520)
x = Tensor.randn(N, n).half().shard(devs, 0).realize()
def f32(x): return host_allgather(x).contiguous()
def f16(x): return host_allgather_half(x).contiguous()
a, b = f32(x), f16(x)
print('max diff', max((Tensor(a.uop.mselect(i)) - Tensor(b.uop.mselect(i))).abs().max().item() for i in range(N)))
for name, fn in [('f32', f32), ('f16', f16)]:
  jit = TinyJit(lambda x: Tensor.stack(*[fn((x * (i + 1)).contiguous()) for i in range(16)]).contiguous())
  for _ in range(3): jit(x)
  for d in devs: Device[d].synchronize()
  st = time.perf_counter()
  for _ in range(5): jit(x)
  for d in devs: Device[d].synchronize()
  print(name, f'{(time.perf_counter()-st)/5/16*1e6:.1f} us each')
