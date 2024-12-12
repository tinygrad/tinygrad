# basic self-contained tests of the external functionality of tinygrad
import unittest, random
from tinygrad import Tensor, Context, Variable, TinyJit, dtypes, Device
from tinygrad.helpers import IMAGE

class TestTiny(unittest.TestCase):

  # *** basic functionality ***

  def test_plus(self):
    out = Tensor([1.,2,3]) + Tensor([4.,5,6])
    self.assertListEqual(out.tolist(), [5.0, 7.0, 9.0])

  def test_plus_big(self):
    out = Tensor.ones(16).contiguous() + Tensor.ones(16).contiguous()
    self.assertListEqual(out.tolist(), [2]*16)

  def test_cat(self):
    out = Tensor.cat(Tensor.ones(8).contiguous(), Tensor.ones(8).contiguous())
    self.assertListEqual(out.tolist(), [1]*16)

  def test_sum(self):
    out = Tensor.ones(256).contiguous().sum()
    self.assertEqual(out.item(), 256)

  def test_gemm(self, N=4, out_dtype=dtypes.float):
    a = Tensor.ones(N,N).contiguous()
    b = Tensor.eye(N).contiguous()
    self.assertListEqual((out:=a@b).flatten().tolist(), [1.0]*(N*N))
    if IMAGE < 2: self.assertEqual(out.dtype, out_dtype)

  # *** randomness ***

  def test_random(self):
    out = Tensor.rand(10)
    for x in out.tolist():
      self.assertGreaterEqual(x, 0.0)
      self.assertLessEqual(x, 1.0)

  # *** JIT (for Python speed) ***

  def test_jit(self):
    cnt = 0
    random.seed(0)
    def new_rand_list(ln=10): return [random.randint(0, 100000) for _ in range(ln)]

    @TinyJit
    def fxn(a,b) -> Tensor:
      nonlocal cnt
      cnt += 1
      return a+b

    for _ in range(3):
      la,lb = new_rand_list(), new_rand_list()
      fa,fb = Tensor(la), Tensor(lb)
      ret = fxn(fa, fb)
      # math is correct
      self.assertListEqual(ret.tolist(), [a+b for a,b in zip(la, lb)])

    # function is only called twice
    self.assertEqual(cnt, 2)

  # *** BEAM (for Kernel speed) ***

  def test_beam(self):
    with Context(BEAM=1): self.test_plus()

  # *** symbolic (to allow less recompilation) ***

  def test_symbolic(self):
    i = Variable('i', 1, 10)
    for s in [2,5]:
      ret = Tensor.ones(s).contiguous().reshape(i.bind(s)) + 1
      self.assertListEqual(ret.reshape(s).tolist(), [2.0]*s)

  def test_symbolic_reduce(self):
    i = Variable('i', 1, 10)
    for s in [2,5]:
      ret = Tensor.ones(s).contiguous().reshape(i.bind(s)).sum()
      self.assertEqual(ret.item(), s)

  # *** image ***

  @unittest.skipIf(Device.DEFAULT != "GPU", "image only supported on GPU")
  def test_image(self):
    with Context(IMAGE=2): self.test_gemm(out_dtype=dtypes.imagef((4, 1, 4)))

  def test_beam_image(self):
    with Context(BEAM=1): self.test_image()

if __name__ == '__main__':
  unittest.main()

