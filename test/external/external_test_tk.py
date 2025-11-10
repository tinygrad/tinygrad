import unittest

from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.engine.realize import ExecItem, get_runner

from extra.thunder.tiny.tk import WARP_THREADS
from extra.thunder.tiny.tk.kernel import Kernel
from extra.thunder.tiny.tk.tiles import gl, st, rt, rv

class TestTK(unittest.TestCase):
  @unittest.skip("store from float rt is wrong")
  def test_simple_matmul(self):
    N = 32
    BLOCK_SIZE = 16
    with Kernel((N // BLOCK_SIZE, N // BLOCK_SIZE, 1), WARP_THREADS) as ker:
      warp = ker.warp

      c = gl((1, 1, N, N), dtypes.float32)
      a = gl((1, 1, N, N), dtypes.bfloat16)
      b = gl((1, 1, N, N), dtypes.bfloat16)

      a_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      b_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      c_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      b_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      c_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      c_reg = warp.zero(c_reg)
      for tile in ker.range(N // BLOCK_SIZE):
        a_smem = warp.load(a_smem, a, (), (0, 0, row, tile), axis=2)
        b_smem = warp.load(b_smem, b, (), (0, 0, tile, col), axis=2)

        a_reg = warp.load(a_reg, a_smem)
        b_reg = warp.load(b_reg, b_smem, transpose=True)

        c_reg = warp.mma_AB(c_reg, a_reg, b_reg)
      c_reg = ker.endrange()

      c_smem = warp.store(c_smem, c_reg)
      c = warp.store(c, c_smem, (0, 0, row, col), (), axis=2)

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

    assert ref.allclose(c)

  @unittest.skip("store from float rt is wrong")
  def test_simple_matmul_transposed(self):
    N = 32
    BLOCK_SIZE = 16
    with Kernel((N // BLOCK_SIZE, N // BLOCK_SIZE, 1), WARP_THREADS) as ker:
      warp = ker.warp

      c = gl((1, 1, N, N), dtypes.float32)
      a = gl((1, 1, N, N), dtypes.bfloat16)
      b = gl((1, 1, N, N), dtypes.bfloat16)

      a_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      b_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      c_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      b_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      c_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      c_reg = warp.zero(c_reg)
      for tile in ker.range(N // BLOCK_SIZE):
        a_smem = warp.load(a_smem, a, (), (0, 0, row, tile), axis=2)
        b_smem = warp.load(b_smem, b, (), (0, 0, col, tile), axis=2)

        a_reg = warp.load(a_reg, a_smem)
        b_reg = warp.load(b_reg, b_smem)

        c_reg = warp.mma_ABt(c_reg, a_reg, b_reg)
      c_reg = ker.endrange()

      c_smem = warp.store(c_smem, c_reg)
      c = warp.store(c, c_smem, (0, 0, row, col), (), axis=2)

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

    assert ref.allclose(c)

  def test_load_store(self):
    N = 32
    BLOCK_SIZE = 16
    with Kernel((N // BLOCK_SIZE, N // BLOCK_SIZE, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = gl((1, 1, N, N), dtypes.float32)
      a = gl((1, 1, N, N), dtypes.float32)

      a_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)
      b_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)
      b_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      a_smem = warp.load(a_smem, a, (), (0, 0, row, col), axis=2)
      a_reg = warp.load(a_reg, a_smem)
      b_reg = warp.copy(b_reg, a_reg)
      b_smem = warp.store(b_smem, b_reg)
      b = warp.store(b, b_smem, (0, 0, row, col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (b, a)])
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float()

    assert ref.allclose(b)

  def test_max(self):
    N = 16
    BLOCK_SIZE = 16
    with Kernel((1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = gl((1, 1, N, N), dtypes.float32)
      a = gl((1, 1, N, N), dtypes.float32)

      a_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)
      b_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)
      b_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      max_reg = rv(BLOCK_SIZE, dtypes.float32, "ortho")

      max_reg = warp.neg_inf(max_reg)

      for tile_row in ker.range(N // BLOCK_SIZE):
        for tile_col in ker.range(N // BLOCK_SIZE):
          a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, tile_col), axis=2)
          a_reg = warp.load(a_reg, a_smem)
          max_reg = warp.row_reduce(max_reg, a_reg, lambda a, b: a.maximum(b))
        sum_reg = ker.endrange()

        b_reg = warp.zero(b_reg).after(tile_row)
        b_reg = warp.map(b_reg, lambda _, idx: sum_reg[idx[0], 0, (idx[2]%4)//2])
        b_smem = warp.store(b_smem, b_reg)

        for tile_col in ker.range(N // BLOCK_SIZE):
          b = warp.store(b, b_smem, (0, 0, tile_row, tile_col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (b, a)])
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().max(axis=3, keepdim=True).expand(a.shape)

    assert ref.allclose(b)

  def test_max_nonsquare(self):
    N, M = 16, 64
    BLOCK_N, BLOCK_M = 16, 64
    with Kernel((1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = gl((1, 1, N, M), dtypes.float32)
      a = gl((1, 1, N, M), dtypes.float32)

      a_smem = st((BLOCK_N, BLOCK_M), dtypes.float32)
      b_smem = st((BLOCK_N, BLOCK_M), dtypes.float32)

      a_reg = rt((BLOCK_N, BLOCK_M), dtypes.float32)
      b_reg = rt((BLOCK_N, BLOCK_M), dtypes.float32)

      max_reg = rv(BLOCK_N, dtypes.float32, "ortho")

      max_reg = warp.zero(max_reg)

      for tile_row in ker.range(N // BLOCK_N):
        for tile_col in ker.range(M // BLOCK_M):
          a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, tile_col), axis=2)
          a_reg = warp.load(a_reg, a_smem)
          sum_reg = warp.row_reduce(max_reg, a_reg, lambda a, b: a.maximum(b))
        sum_reg = ker.endrange()

        b_reg = warp.zero(b_reg).after(tile_row)
        b_reg = warp.map(b_reg, lambda _, idx: sum_reg[idx[0], 0, (idx[2]%4)//2])
        b_smem = warp.store(b_smem, b_reg)

        for tile_col in ker.range(M // BLOCK_M):
          b = warp.store(b, b_smem, (0, 0, tile_row, tile_col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, M, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, M, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (b, a)])
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().max(axis=3, keepdim=True).expand(a.shape)

    assert ref.allclose(b)

  def test_sum(self):
    N = 16
    BLOCK_SIZE = 16
    with Kernel((1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = gl((1, 1, N, N), dtypes.float32)
      a = gl((1, 1, N, N), dtypes.float32)

      a_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)
      b_smem = st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)
      b_reg = rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      sum_reg = rv(BLOCK_SIZE, dtypes.float32, "ortho")

      for tile_row in ker.range(N // BLOCK_SIZE):
        sum_reg = warp.zero(sum_reg).after(tile_row)

        for tile_col in ker.range(N // BLOCK_SIZE):
          a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, tile_col), axis=2)
          a_reg = warp.load(a_reg, a_smem)
          sum_reg = warp.row_reduce(sum_reg, a_reg, lambda a, b: a + b)
        sum_reg = ker.endrange()

        b_reg = warp.zero(b_reg).after(tile_row)
        b_reg = warp.map(b_reg, lambda _, idx: sum_reg[idx[0], 0, (idx[2]%4)//2])
        b_smem = warp.store(b_smem, b_reg)

        for tile_col in ker.range(N // BLOCK_SIZE):
          b = warp.store(b, b_smem, (0, 0, tile_row, tile_col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
      a = Tensor.arange(1 * 1 * N * N).reshape(1, 1, N, N).cast(dtypes.float32).contiguous()
      b = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (b, a)])
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().sum(axis=3, keepdim=True).expand(a.shape)

    assert ref.allclose(b)

  def test_sum_nonsquare(self):
    N, M = 16, 64
    BLOCK_N, BLOCK_M = 16, 64
    with Kernel((1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = gl((1, 1, N, M), dtypes.float32)
      a = gl((1, 1, N, M), dtypes.float32)

      a_smem = st((BLOCK_N, BLOCK_M), dtypes.float32)
      b_smem = st((BLOCK_N, BLOCK_M), dtypes.float32)

      a_reg = rt((BLOCK_N, BLOCK_M), dtypes.float32)
      b_reg = rt((BLOCK_N, BLOCK_M), dtypes.float32)

      sum_reg = rv(BLOCK_N, dtypes.float32, "ortho")

      sum_reg = warp.zero(sum_reg)

      for tile_row in ker.range(N // BLOCK_N):
        for tile_col in ker.range(M // BLOCK_M):
          a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, tile_col), axis=2)
          a_reg = warp.load(a_reg, a_smem)
          sum_reg = warp.row_reduce(sum_reg, a_reg, lambda a, b: a + b)
        sum_reg = ker.endrange()

        b_reg = warp.zero(b_reg).after(tile_row)
        b_reg = warp.map(b_reg, lambda _, idx: sum_reg[idx[0], 0, (idx[2]%4)//2])
        b_smem = warp.store(b_smem, b_reg)

        for tile_col in ker.range(M // BLOCK_M):
          b = warp.store(b, b_smem, (0, 0, tile_row, tile_col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, M, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, M, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in (b, a)])
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().sum(axis=3, keepdim=True).expand(a.shape)

    assert ref.allclose(b)

if __name__ == "__main__":
  unittest.main()
