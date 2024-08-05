import unittest
import numpy as np
from tinygrad import Tensor, GlobalCounters, dtypes
from tinygrad.helpers import Context, getenv
from tinygrad.engine.realize import run_schedule
from tinygrad.codegen.kernel import Opt, OptOps, Kernel
from tinygrad.engine.realize import CompiledRunner, ExecItem

class TestArange(unittest.TestCase):
  def _get_flops(self, N, opts=None):
    GlobalCounters.reset()
    tt = Tensor.arange(N)
    sched = tt.schedule()
    self.assertEqual(len(sched), 1)
    k = Kernel(sched[-1].ast)
    if opts is not None:
      for o in opts: k.apply_opt(o)
    p = k.to_program()
    print(p.name)
    print(p.src)
    ExecItem(CompiledRunner(p), [tt.lazydata.buffer]).run()
    np.testing.assert_equal(tt.numpy(), np.arange(N))
    return p.op_estimate

  def test_complexity(self, opts=None):
    # add 1 to avoid divide by 0. arange is 0 flops now!
    f1 = self._get_flops(256, opts) + 1
    f2 = self._get_flops(2560, opts) + 1
    print(f"{f1=}, {f2=}")
    assert f2 / f1 < 15, f"bad complexity, flops {f2/f1:.1f}X while inputs 10X"

  def test_complexity_w_upcast(self): return self.test_complexity([Opt(OptOps.UPCAST, 0, 4)])
  def test_complexity_w_unroll(self): return self.test_complexity([Opt(OptOps.UNROLL, 0, 4)])
  def test_complexity_w_upcast_and_unroll(self): return self.test_complexity([Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 4)])

class TestIndexing(unittest.TestCase):
  def test_arange_2_reduce(self):
    needle = Tensor.zeros(16384, dtype=dtypes.int).contiguous()
    needle[1337] = 1
    needle.realize()
    with Context(NOOPT=1, FUSE_ARANGE=1):
      GlobalCounters.reset()
      # TODO: it should work without these reshapes
      out = ((Tensor.arange(1,16385).reshape(16384,1)-1)*needle.reshape(16384,1)).sum()
      sched = out.schedule()
      assert len(sched) == 1
      run_schedule(sched)
    assert out.item() == 1337, f"expected 1337, got {out.item()}"

  @unittest.skipIf(getenv("PTX"), "broken on ptx for some reason")
  def test_manual_index(self):
    dataset = Tensor.rand(16384, 256).realize()
    idxs = Tensor([0,3,5,6]).realize()
    real_index = dataset.numpy()[idxs.numpy()]
    print("*** indexing ***")
    with Context(NOOPT=1, FUSE_ARANGE=1):
      GlobalCounters.reset()
      rng = Tensor.ones(4, 256, 16384, dtype=dtypes.int)._cumsum(axis=-1, _first_zero=True).reshape(4, 256, 16384, 1)
      idxs = idxs.reshape(4,1,1,1).expand(4, 256, 16384, 1)
      reshape_dataset = dataset.T.reshape(1, 256, 16384, 1).expand(4, 256, 16384, 1)
      full = (rng==idxs).where(reshape_dataset, Tensor.zeros(4, 256, 16384, 1))
      X = full.sum(axis=(2,3))
      sched = X.schedule()
      assert len(sched) == 1
      run_schedule(sched)
      assert GlobalCounters.global_ops < 4*16384, f"too many ops {GlobalCounters.global_ops}"
    np.testing.assert_allclose(real_index, X.numpy())

  def test_index(self):
    dataset = Tensor.rand(16384, 256).realize()
    idxs = Tensor([0,3,5,6]).realize()
    real_index = dataset.numpy()[idxs.numpy()]
    print("*** indexing ***")
    with Context(NOOPT=1):
      GlobalCounters.reset()
      X = dataset[idxs]
      assert X.shape == (4,256)
      sched = X.schedule()
      # TODO: enable these asserts when the scheduler can handle this
      #assert len(sched) == 1, f"{len(sched)} != 1"
      run_schedule(sched)
      #assert GlobalCounters.global_ops < 4*16384, f"too many ops {GlobalCounters.global_ops}"
    np.testing.assert_allclose(real_index, X.numpy())

  def test_index_fused(self, noopt=1):
    dataset = Tensor.rand(16384, 256).realize()
    idxs = Tensor([0,3,5,6]).realize()
    real_index = dataset.numpy()[idxs.numpy()]
    print("*** indexing ***")
    with Context(NOOPT=noopt, FUSE_ARANGE=1):
      GlobalCounters.reset()
      X = dataset[idxs]
      assert X.shape == (4,256)
      sched = X.schedule()
      assert len(sched) == 2
      run_schedule(sched)
      assert GlobalCounters.global_ops < 4*16384, f"too many ops {GlobalCounters.global_ops} != {4*16384}"
    np.testing.assert_allclose(real_index, X.numpy())
  @unittest.skip("not ready")
  def test_index_fused_opt(self): self.test_index_fused(0)

  @unittest.skipIf(getenv("PTX"), "broken on ptx for some reason")
  def test_index_mnist(self, noopt=1):
    from tinygrad.nn.datasets import mnist
    X_train, Y_train, _, _ = mnist()
    with Context(NOOPT=noopt, FUSE_ARANGE=1, SPLIT_REDUCEOP=0):
      GlobalCounters.reset()
      samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
      x = X_train[samples].numpy()
      y = Y_train[samples].numpy()
      assert GlobalCounters.global_ops < 4*16384, f"too many ops {GlobalCounters.global_ops} != {4*16384}"
    np.testing.assert_allclose(X_train.numpy()[samples.numpy()], x)
    np.testing.assert_allclose(Y_train.numpy()[samples.numpy()], y)
  @unittest.skip("not ready")
  def test_index_mnist_opt(self): self.test_index_mnist(0)

if __name__ == "__main__":
  unittest.main()
