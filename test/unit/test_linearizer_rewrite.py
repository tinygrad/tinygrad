import unittest
from tinygrad import Tensor, Context, Device
from tinygrad.codegen.kernel import Kernel, Opt, OptOps

class TestLinearizerRewrite(unittest.TestCase):
  def test_reduction(self):
    t = Tensor.ones((64,64), device="NULL").contiguous().realize()
    out = (t*2).sum(axis=1)
    with Context(SPLIT_REDUCEOP=0, DEVECTORIZE=0):
      si = out.schedule()[-1]
      k = Kernel(si.ast, Device["CPU"].renderer)
      #k.apply_opt(Opt(OptOps.UPCAST, 0, 4))
      k.apply_opt(Opt(OptOps.UNROLL, 0, 4))
      prg = k.to_program()
      # acc0 = (acc0+(val0[0]*2.0f)+(val0[1]*2.0f)+(val0[2]*2.0f)+(val0[3]*2.0f));
      print(prg.src)

if __name__ == '__main__':
  unittest.main()
