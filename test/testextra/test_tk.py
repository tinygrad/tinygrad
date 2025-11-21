import unittest, math, time

from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.helpers import CI
import numpy as np

from extra.thunder.tiny.tk import WARP_THREADS
from extra.thunder.tiny.tk.kernel import Kernel
from extra.thunder.tiny.tk.tiles import ST_16X32, RT_16X32, RT_16X16, TileLayout

@unittest.skipIf(CI and Device.DEFAULT not in ["AMD"], "only amd")
class TestTK(unittest.TestCase):
  @unittest.skipIf(CI, "no wmma in ci")
  def test_simple_matmul(self):
    N = 8192
    BLOCK_SIZE = 64
    with Kernel((N // BLOCK_SIZE, N // BLOCK_SIZE, 1), WARP_THREADS) as ker:
      warp = ker.warp

      c = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.bfloat16)
      b = ker.gl((1, 1, N, N), dtypes.bfloat16)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      b_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      b_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      c_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      c_reg = warp.zero(c_reg)
      for tile in ker.range(N // BLOCK_SIZE):
        a_smem = warp.load(a_smem, a, (), (0, 0, row, tile), axis=2)
        b_smem = warp.load(b_smem, b, (), (0, 0, tile, col), axis=2)

        a_reg = warp.load(a_reg, a_smem)
        b_reg = warp.load(b_reg, b_smem)

        c_reg = warp.mma_AB(c_reg, a_reg, b_reg)
      c_reg = ker.endrange()

      c = warp.store(c, c_reg, (0, 0, row, col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="bfloat16").contiguous()
      b = Tensor.rand(1, 1, N, N, dtype="bfloat16").contiguous()
      c = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b, c)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (c, a, b)])
    for _ in range(5): ei.run(wait=True)
    c = c.float()

    ref = a.matmul(b, dtype=dtypes.float32).float()

    np.testing.assert_allclose(c.numpy(), ref.numpy())

  @unittest.skipIf(CI, "no wmma in ci")
  def test_simple_matmul_transposed(self):
    N = 8192
    BLOCK_N, BLOCK_M, BLOCK_K = 64, 64, 128
    with Kernel((N // BLOCK_N, N // BLOCK_M, 1), WARP_THREADS) as ker:
      warp = ker.warp

      c = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.bfloat16)
      b = ker.gl((1, 1, N, N), dtypes.bfloat16)

      a_smem = ker.st((BLOCK_N, BLOCK_K), dtypes.bfloat16, base_shape=ST_16X32)
      b_smem = ker.st((BLOCK_M, BLOCK_K), dtypes.bfloat16, base_shape=ST_16X32)

      a_reg = ker.rt((BLOCK_N, BLOCK_K), dtypes.bfloat16, base_shape=RT_16X32)
      b_reg = ker.rt((BLOCK_M, BLOCK_K), dtypes.bfloat16, base_shape=RT_16X32)
      c_reg = ker.rt((BLOCK_N, BLOCK_M), dtypes.float32, TileLayout.COL, base_shape=RT_16X16)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      c_reg = warp.zero(c_reg)
      for tile in ker.range(N // BLOCK_K):
        a_smem = warp.load(a_smem, a, (), (0, 0, row, tile), axis=2)
        b_smem = warp.load(b_smem, b, (), (0, 0, col, tile), axis=2)

        a_reg = warp.load(a_reg, a_smem)
        b_reg = warp.load(b_reg, b_smem)

        c_reg = warp.mma_ABt(c_reg, a_reg, b_reg)
      c_reg = ker.endrange()

      c = warp.store(c, c_reg, (0, 0, row, col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="bfloat16").contiguous()
      b = Tensor.rand(1, 1, N, N, dtype="bfloat16").contiguous()
      c = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b, c)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (c, a, b)])
    for _ in range(5): ei.run(wait=True)
    c = c.float()

    ref = a.matmul(b.transpose(2, 3), dtype=dtypes.float32).float()

    np.testing.assert_allclose(c.numpy(), ref.numpy())

  def test_load_store(self):
    N = 64
    BLOCK_SIZE = 32
    with Kernel((N // BLOCK_SIZE, N // BLOCK_SIZE, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)
      b_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      a_smem = warp.load(a_smem, a, (), (0, 0, row, col), axis=2)
      a_reg = warp.load(a_reg, a_smem)
      b_reg = warp.copy(b_reg, a_reg)
      b = warp.store(b, b_reg, (0, 0, row, col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (b, a)])
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float()

    np.testing.assert_allclose(b.numpy(), ref.numpy())

  @unittest.skip("TODO")
  def test_load_store_group(self):
    N = 256
    BLOCK_SIZE = 64
    with Kernel((N // BLOCK_SIZE, N // BLOCK_SIZE, 1), WARP_THREADS * 2) as ker:
      warp = ker.warp
      group = ker.group(2)

      b = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)
      b_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      a_smem = group.load(a_smem, a, (), (0, 0, row, col), axis=2)
      a_reg = warp.load(a_reg, a_smem)
      b_reg = warp.copy(b_reg, a_reg)
      b = warp.store(b, b_reg, (0, 0, row, col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (b, a)])
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float()

    np.testing.assert_allclose(b.numpy(), ref.numpy())

  def test_add(self):
    N = 64
    BLOCK_SIZE = 32
    with Kernel((1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      for tile_row in ker.range(N // BLOCK_SIZE):
        for tile_col in ker.range(N // BLOCK_SIZE):
          a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, tile_col), axis=2)
          a_reg = warp.load(a_reg, a_smem)

          a_reg += 1

          b = warp.store(b, a_reg, (0, 0, tile_row, tile_col), (), axis=2)

      sink = ker.finish()

      with Context(DEBUG=0):
        a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
        b = Tensor.empty(1, 1, N, N, dtype="float32")
        Tensor.realize(a, b)

      ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (b, a)])
      for _ in range(5): ei.run(wait=True)
      b = b.float()

      ref = a.float() + 1

      np.testing.assert_allclose(b.numpy(), ref.numpy())

  def test_max(self):
    N = 64
    BLOCK_SIZE = 32
    with Kernel((1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      b_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      max_reg = ker.rv(BLOCK_SIZE, dtypes.float32)

      for tile_col in ker.range(N // BLOCK_SIZE):
        max_reg = warp.neg_inf(max_reg.after(tile_col))

        for tile_row in ker.range(N // BLOCK_SIZE):
          a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, tile_col), axis=2)
          a_reg = warp.load(a_reg, a_smem)
          max_reg = warp.col_reduce(max_reg, a_reg, lambda a, b: a.maximum(b), init_value=-math.inf)
        max_reg = ker.endrange()

        b_reg = warp.map(b_reg, lambda _, idx: max_reg[idx[1], 0])

        for tile_row in ker.range(N // BLOCK_SIZE):
          b = warp.store(b, b_reg, (0, 0, tile_row, tile_col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (b, a)])
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().max(axis=2, keepdim=True).expand(a.shape)

    np.testing.assert_allclose(b.numpy(), ref.numpy())

  def test_max_nonsquare(self):
    N, M = 32, 128
    BLOCK_N, BLOCK_M = 16, 64
    with Kernel((1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, M), dtypes.float32)
      a = ker.gl((1, 1, N, M), dtypes.float32)

      a_smem = ker.st((BLOCK_N, BLOCK_M), dtypes.float32)

      a_reg = ker.rt((BLOCK_N, BLOCK_M), dtypes.float32, TileLayout.COL)
      b_reg = ker.rt((BLOCK_N, BLOCK_M), dtypes.float32, TileLayout.COL)

      max_reg = ker.rv(BLOCK_M, dtypes.float32)

      for tile_col in ker.range(M // BLOCK_M):
        max_reg = warp.neg_inf(max_reg.after(tile_col))

        for tile_row in ker.range(N // BLOCK_N):
          a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, tile_col), axis=2)
          a_reg = warp.load(a_reg, a_smem)
          max_reg = warp.col_reduce(max_reg, a_reg, lambda a, b: a.maximum(b), init_value=-math.inf)
        max_reg = ker.endrange()

        b_reg = warp.map(b_reg, lambda _, idx: max_reg[idx[1], 0])

        for tile_row in ker.range(N // BLOCK_N):
          b = warp.store(b, b_reg, (0, 0, tile_row, tile_col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, M, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, M, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (b, a)])
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().max(axis=2, keepdim=True).expand(a.shape)

    np.testing.assert_allclose(b.numpy(), ref.numpy())

  def test_sum(self):
    N = 64
    BLOCK_SIZE = 32
    with Kernel((1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      b_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      sum_reg = ker.rv(BLOCK_SIZE, dtypes.float32)

      for tile_col in ker.range(N // BLOCK_SIZE):
        sum_reg = warp.zero(sum_reg.after(tile_col))

        for tile_row in ker.range(N // BLOCK_SIZE):
          a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, tile_col), axis=2)
          a_reg = warp.load(a_reg, a_smem)
          sum_reg = warp.col_reduce(sum_reg, a_reg, lambda a, b: a + b)
        sum_reg = ker.endrange()

        b_reg = warp.map(b_reg, lambda _, idx: sum_reg[idx[1], 0])

        for tile_row in ker.range(N // BLOCK_SIZE):
          b = warp.store(b, b_reg, (0, 0, tile_row, tile_col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (b, a)])
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().sum(axis=2, keepdim=True).expand(a.shape)

    np.testing.assert_allclose(b.numpy(), ref.numpy(), atol=1e-5, rtol=1e-5)

  def test_sum_nonsquare(self):
    N, M = 32, 128
    BLOCK_N, BLOCK_M = 16, 64
    with Kernel((1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, M), dtypes.float32)
      a = ker.gl((1, 1, N, M), dtypes.float32)

      a_smem = ker.st((BLOCK_N, BLOCK_M), dtypes.float32)

      a_reg = ker.rt((BLOCK_N, BLOCK_M), dtypes.float32, TileLayout.COL)
      b_reg = ker.rt((BLOCK_N, BLOCK_M), dtypes.float32, TileLayout.COL)

      sum_reg = ker.rv(BLOCK_M, dtypes.float32)

      for tile_col in ker.range(M // BLOCK_M):
        sum_reg = warp.zero(sum_reg.after(tile_col))

        for tile_row in ker.range(N // BLOCK_N):
          a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, tile_col), axis=2)
          a_reg = warp.load(a_reg, a_smem)
          sum_reg = warp.col_reduce(sum_reg, a_reg, lambda a, b: a + b)
        sum_reg = ker.endrange()

        b_reg = warp.map(b_reg, lambda _, idx: sum_reg[idx[1], 0])

        for tile_row in ker.range(N // BLOCK_N):
          b = warp.store(b, b_reg, (0, 0, tile_row, tile_col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, M, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, M, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (b, a)])
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().sum(axis=2, keepdim=True).expand(a.shape)

    np.testing.assert_allclose(b.numpy(), ref.numpy(), atol=1e-5, rtol=1e-5)

  # @unittest.skip("fake range not ended")
  def test_softmax(self):
    N = 64
    BLOCK_SIZE = 32
    with Kernel((1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, BLOCK_SIZE, N), dtypes.float32)
      a = ker.gl((1, 1, BLOCK_SIZE, N), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      max_vec_last = ker.rv(BLOCK_SIZE, dtypes.float32)
      max_vec = ker.rv(BLOCK_SIZE, dtypes.float32)
      norm_vec = ker.rv(BLOCK_SIZE, dtypes.float32)

      max_vec = warp.neg_inf(max_vec)
      norm_vec = warp.zero(norm_vec)

      for tile_col in ker.range(N // BLOCK_SIZE):
        a_smem = warp.load(a_smem, a, (), (0, 0, 0, tile_col), axis=2)
        a_reg = warp.load(a_reg, a_smem)

        a_reg *= 1.0 / math.log(2)

        max_vec_last = warp.copy(max_vec_last.after(tile_col), max_vec)
        max_vec = warp.row_reduce(max_vec.after(max_vec_last), a_reg, lambda a, b: a.maximum(b), init_value=-math.inf)
        a_reg = (a_reg - max_vec).exp2()
        max_vec_last = (max_vec_last - max_vec).exp2()
        norm_vec *= max_vec_last
        norm_vec = warp.row_reduce(norm_vec, a_reg, lambda a, b: a + b)
      norm_vec = ker.endrange()

      for tile_col in ker.range(N // BLOCK_SIZE):
        a_smem = warp.load(a_smem, a, (), (0, 0, 0, tile_col), axis=2)
        a_reg = warp.load(a_reg.after(norm_vec), a_smem)

        a_reg *= 1.0 / math.log(2)
        a_reg = (a_reg - max_vec).exp2()
        a_reg /= norm_vec

        b = warp.store(b, a_reg, (0, 0, 0, tile_col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      Tensor.manual_seed(42)
      a = Tensor.rand(1, 1, BLOCK_SIZE, N, dtype="float32")
      b = Tensor.empty(1, 1, BLOCK_SIZE, N, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (b, a)])
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().softmax(axis=3)

    np.testing.assert_allclose(b.numpy(), ref.numpy(), atol=1e-5, rtol=1e-5)

  def test_softmax_col(self):
    N = 64
    BLOCK_SIZE = 32
    with Kernel((1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, BLOCK_SIZE), dtypes.float32)
      a = ker.gl((1, 1, N, BLOCK_SIZE), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      max_vec_last = ker.rv(BLOCK_SIZE, dtypes.float32)
      max_vec = ker.rv(BLOCK_SIZE, dtypes.float32)
      norm_vec = ker.rv(BLOCK_SIZE, dtypes.float32)

      max_vec = warp.neg_inf(max_vec)
      norm_vec = warp.zero(norm_vec)

      for tile_row in ker.range(N // BLOCK_SIZE):
        a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, 0), axis=2)
        a_reg = warp.load(a_reg, a_smem)

        a_reg *= 1.0 / math.log(2)

        max_vec_last = warp.copy(max_vec_last.after(tile_row), max_vec)
        max_vec = warp.col_reduce(max_vec.after(max_vec_last), a_reg, lambda a, b: a.maximum(b), init_value=-math.inf)
        a_reg = (a_reg - max_vec).exp2()
        max_vec_last = (max_vec_last - max_vec).exp2()
        norm_vec *= max_vec_last
        norm_vec = warp.col_reduce(norm_vec, a_reg, lambda a, b: a + b)
      norm_vec = ker.endrange()

      for tile_row in ker.range(N // BLOCK_SIZE):
        a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, 0), axis=2)
        a_reg = warp.load(a_reg.after(norm_vec), a_smem)

        a_reg *= 1.0 / math.log(2)
        a_reg = (a_reg - max_vec).exp2()
        a_reg /= norm_vec

        b = warp.store(b, a_reg, (0, 0, tile_row, 0), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      Tensor.manual_seed(42)
      a = Tensor.rand(1, 1, N, BLOCK_SIZE, dtype="float32")
      b = Tensor.empty(1, 1, N, BLOCK_SIZE, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (b, a)])
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().softmax(axis=2)

    np.testing.assert_allclose(b.numpy(), ref.numpy(), atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
  unittest.main()
