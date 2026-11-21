#!/usr/bin/env python3
# DEV=AMD PYTHONPATH=. python3 extra/k3/test_hostwait.py: a graph tells the host a result is ready (store) and waits for the host (wait)
import threading, time, unittest
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.device import Buffer, BufferSpec
from tinygrad.uop.ops import UOp
from extra.k3.host import store, wait

def host_buffer(n:int) -> tuple[Tensor, np.ndarray]:
  buf = Buffer(Device.DEFAULT, 4 * n, options=BufferSpec(host=True, uncached=True, cpu_access=True)).ensure_allocated()
  (arr:=np.frombuffer(buf.host.view(fmt='B').mv, np.uint32))[:] = 0
  return Tensor(UOp.from_buffer(buf, dtypes.uint32)), arr

def step(y:Tensor, tag:Tensor, table:Tensor) -> Tensor: # tag after y, then y + the 16 slots of the table [ready | slots] once the host is done
  table = table.uop.after(wait(table.uop.after(store(tag.uop, y.uop))))
  return y + Tensor(table)[1:].float()

class TestHostWait(unittest.TestCase):
  def test_roundtrip(self):
    (tag, tag_np), (table, table_np) = host_buffer(1), host_buffer(17)
    y = Tensor(np.arange(16, dtype=np.float32) * 2).realize()
    z = step(y, tag, table).realize() # queued: the gpu waits for the host
    def host(): # sees y done, fills the table, releases the gpu
      while tag_np[0] != 1: pass
      time.sleep(0.2)
      table_np[1:] = 7 + np.arange(16)
      table_np[0] = 1
    (t:=threading.Thread(target=host)).start()
    st = time.perf_counter()
    out = z.numpy()
    t.join()
    self.assertGreaterEqual(time.perf_counter() - st, 0.1)
    self.assertEqual(out.tolist(), [3 * i + 7 for i in range(16)])

  def test_jit(self): # every replay waits again: the host clears the flags before it runs the graph
    (tag, tag_np), (table, table_np) = host_buffer(1), host_buffer(17)
    f = TinyJit(lambda y: step((y * 2).contiguous(), tag, table).realize())
    for it in range(5):
      tag_np[0] = table_np[0] = 0
      z = f(Tensor(np.full(16, it, dtype=np.float32)).realize())
      while tag_np[0] != 1: pass
      table_np[1:] = 100 * it + np.arange(16)
      table_np[0] = 1
      self.assertEqual(z.numpy().tolist(), [2 * it + 100 * it + i for i in range(16)])

if __name__ == "__main__":
  unittest.main()
