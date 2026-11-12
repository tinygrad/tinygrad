import unittest
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.engine import jit

class TestPreforkForget(unittest.TestCase):
  def test_forget_bases_views_multi(self):
    keep, drop = Buffer("NULL", 1024).ensure_allocated(), Buffer("NULL", 1024).ensure_allocated()
    view = drop.view(256, 256)
    (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = [drop.view(128, 0)]
    reg = {id(x): x for x in (keep, drop, view, mb)}
    jit._prefork_regs.append(reg)
    try: jit.prefork_forget([drop])
    finally: jit._prefork_regs.remove(reg)
    self.assertEqual(list(reg.values()), [keep])

  def test_registry_dropped_on_error(self):
    n = len(jit._prefork_regs)
    def boom(*a, **k):
      jit._prefork_regs.append({})
      raise RuntimeError("boom")
    orig, jit._capture_parallel = jit._capture_parallel, boom
    try:
      with self.assertRaises(RuntimeError): jit.capture_parallel([])
    finally: jit._capture_parallel = orig
    self.assertEqual(len(jit._prefork_regs), n)

if __name__ == "__main__":
  unittest.main()
