#!/usr/bin/env python3
# DEV=AMD python3 extra/k3/test_hostwait.py: a graph tells the host a result is ready (hostsignal) and waits for the host (hostwait)
import threading, time, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.device import Buffer, BufferSpec
from tinygrad.uop.ops import UOp

def host_buffer(dev:str, n:int, dtype=dtypes.uint32) -> tuple[Tensor, memoryview]:
  buf = Buffer(dev, n * dtype.itemsize, options=BufferSpec(host=True, uncached=True, cpu_access=True)).ensure_allocated()
  host = buf.get_storage().host
  assert host is not None
  mv = host.mv.cast('B').cast('I' if dtype.itemsize == 4 else 'B')
  mv[:] = memoryview(bytes(n * dtype.itemsize)).cast(mv.format)
  return Tensor(UOp.from_buffer(buf, dtype)), mv

class TestHostWait(unittest.TestCase):
  def test_roundtrip(self):
    dev = Device.DEFAULT
    tag, tag_mv = host_buffer(dev, 1)
    ready, ready_mv = host_buffer(dev, 1)
    slots, slots_mv = host_buffer(dev, 16)
    y = Tensor(np.arange(16, dtype=np.float32) * 2, device=dev).realize()
    signal = UOp.custom_function("hostsignal").call(tag.uop, y.uop)
    wait = UOp.custom_function("hostwait").call(ready.uop.after(signal), slots.uop)
    z = (y + Tensor(slots.uop.after(wait)).float()).realize()  # queued: the gpu blocks in hostwait

    def host():  # sees y done, fills the table, releases the gpu
      while tag_mv[0] != 1: pass
      time.sleep(0.2)
      for i in range(16): slots_mv[i] = 7 + i
      ready_mv[0] = 1
    t = threading.Thread(target=host)
    st = time.perf_counter()
    t.start()
    out = z.numpy()
    t.join()
    self.assertGreaterEqual(time.perf_counter() - st, 0.2)
    self.assertEqual(out.tolist(), [3 * i + 7 for i in range(16)])

  def test_jit(self):  # every replay waits again: the host clears the flags before it runs the graph
    dev = Device.DEFAULT
    (tag, tag_mv), (ready, ready_mv), (slots, slots_mv) = host_buffer(dev, 1), host_buffer(dev, 1), host_buffer(dev, 16)
    @TinyJit
    def step(y:Tensor) -> Tensor:
      y = (y * 2).contiguous()
      wait = UOp.custom_function("hostwait").call(ready.uop.after(UOp.custom_function("hostsignal").call(tag.uop, y.uop)), slots.uop)
      return (y + Tensor(slots.uop.after(wait)).float()).realize()
    for it in range(5):
      tag_mv[0] = ready_mv[0] = 0
      z = step(Tensor(np.full(16, it, dtype=np.float32), device=dev).realize())
      while tag_mv[0] != 1: pass
      for i in range(16): slots_mv[i] = 100 * it + i
      ready_mv[0] = 1
      self.assertEqual(z.numpy().tolist(), [2 * it + 100 * it + i for i in range(16)])

if __name__ == "__main__":
  unittest.main()
