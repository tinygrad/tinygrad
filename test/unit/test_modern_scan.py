import unittest
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, AddrSpace

class TestModernScan(unittest.TestCase):
  def test_copy_local(self):
    N = 256
    state = Tensor.empty(N)
    tmp = UOp.placeholder((N,), state.dtype, slot=-1, addrspace=AddrSpace.LOCAL)
    tmp = tmp.after(tmp.store(state.uop))
    state.assign(tmp)
    state.realize()

  """
  def test_scan_gemv(self):
    N = 256
    gemvs = Tensor.empty(3, N, N)
    state = Tensor.empty(N)
    Tensor.realize(gemvs, state)

    #tmp = UOp.placeholder((N,), state.dtype, slot=-1, addrspace=AddrSpace.REG)
    tmp = Tensor.empty(N, dtype=state.dtype).uop
    tmp = tmp.after(tmp.store(state.uop))
    #rng = UOp.range(3, -1)
    #tmp = tmp.after(tmp.store(state.uop, rng))
    #tmp = tmp.after(tmp.store(tmp @ gemvs.uop[rng]).end(rng))
    state.assign(tmp)

    state.realize()
  """

if __name__ == '__main__':
  unittest.main()



