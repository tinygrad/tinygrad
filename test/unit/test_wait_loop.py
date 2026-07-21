import unittest
from tinygrad import Tensor, UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import KernelInfo

def wait_loop_kernel(C:UOp) -> UOp:
  N = 10

  # LOOP is a bound-less loop header: a jump target with no induction variable.
  # the compare and conditional backedge are expanded by the renderers from LOOP/END
  l = UOp.loop(0)

  i = UOp.placeholder((1,), dtypes.int, 0, addrspace=AddrSpace.REG)

  # i = 0
  i = i.after(i[0].store(0))

  # i + 1, read loop-carried through after(l)
  inc = i.after(l)[0].load() + 1

  # i = inc; END(store, l, cond): conditional backedge, loop again while inc < N (do-while)
  # NOTE: the cond uses the computed value, not a reload of the register
  st = i[0].store(inc)
  i = i.after(st.end(l, inc < N))

  return C[0].store(i[0].load()).sink(arg=KernelInfo(name="wait_loop"))

class TestWaitLoop(unittest.TestCase):
  def test_wait_loop(self):
    c = Tensor.empty(1, dtype=dtypes.int)
    c = Tensor.custom_kernel(c, fxn=wait_loop_kernel)[0]
    c.realize()
    self.assertEqual(c.item(), 10)

if __name__ == "__main__": unittest.main()
