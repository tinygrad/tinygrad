import numpy as np
import unittest
from tinygrad import Tensor
from tinygrad.helpers import get_single_element
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.engine.realize import CompiledRunner, ExecItem

class TestOptGemm(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    N = 64
    cls.a = Tensor.randn(N, N).contiguous().realize()
    cls.b = Tensor.randn(N, N).contiguous().realize()

  def _test_gemm_unrolled_permute_l(self, opts=[]):
    t = self.a.T @ self.b.T
    res = self.a.T.numpy() @ self.b.T.numpy()
    # TODO: this should be a generic test helper
    si = get_single_element(t.schedule())
    k = Kernel(si.ast)
    k.apply_opts(opts)
    run = CompiledRunner(k.to_program())
    ExecItem(run, si.bufs).run()
    test = si.bufs[0].numpy().reshape(res.shape)
    np.testing.assert_allclose(res, test, atol=1e-4)

  def test_gemm_unrolled_permute_l_44(self):
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=4)]
    self._test_gemm_unrolled_permute_l(opts)

  def test_gemm_unrolled_permute_l_424(self):
    # was failing with LLVM
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=2), Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    self._test_gemm_unrolled_permute_l(opts)

  def test_gemm_unrolled_permute_l_42(self):
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=2)]
    self._test_gemm_unrolled_permute_l(opts)

if __name__ == '__main__':
  unittest.main()
