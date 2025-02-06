import time
from tinygrad import Tensor, TinyJit, Device, Context
from tinygrad.helpers import Profiling, Timing, GlobalCounters

# python3 test/test_speed_v_torch.py TestSpeed.test_add_a

@TinyJit
def plus(a:Tensor, b:Tensor): return a+b

if __name__ == "__main__":
  a = Tensor([1]).realize()
  b = Tensor([1]).realize()
  for i in range(5):
    with Timing(prefix=f"{i}:"):
      c = plus(a,b)
      Device[c.device].synchronize()
  assert c.item() == 2
  assert GlobalCounters.time_sum_s == 0
  with Context(DEBUG=2):
    st = time.perf_counter()
    c = plus(a,b)
    Device[c.device].synchronize()
    et = time.perf_counter() - st
  print(f"kernel {GlobalCounters.time_sum_s*1e3:.2f} ms / full {et*1e3:.2f} ms -- {et/GlobalCounters.time_sum_s:.2f} x")
  with Profiling():
    c = plus(a,b)
    Device[c.device].synchronize()
