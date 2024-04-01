import unittest
from tinygrad import Tensor, Device
from tinygrad.engine.schedule import create_schedule
from tinygrad.features.multi import MultiLazyBuffer
from tinygrad.helpers import CI
from tinygrad.ops import BufferOps
import numpy as np

def _check_ast_count(desired_count:int, t:Tensor):
  # NOTE: this has side effect because everything can be scheduled only once
  schedule = create_schedule(t.lazydata.lbs if isinstance(t.lazydata, MultiLazyBuffer) else [t.lazydata])
  asts = [s for s in schedule if s.ast[0].op is BufferOps.STORE]
  assert len(asts) == desired_count

class TestSimpleConstFolding(unittest.TestCase):
  def test_add_literal_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) + 0)
  def test_add_tensor_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) + Tensor.zeros(4))
  def test_literal_zero_add(self):
    _check_ast_count(0, 0 + Tensor([1.0, 2, 3, 4]))
  def test_tensor_zero_add(self):
    _check_ast_count(0, Tensor.zeros(4) + Tensor([1.0, 2, 3, 4]))

  def test_sub_literal_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) - 0)
  def test_sub_tensor_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) - Tensor.zeros(4))

  def test_mul_literal_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * 0)
  def test_mul_tensor_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * Tensor.zeros(4))
  def test_literal_zero_mul(self):
    _check_ast_count(0, 0 * Tensor([1.0, 2, 3, 4]) * 0)
  def test_tensor_zero_mul(self):
    _check_ast_count(0, Tensor.zeros(4) * Tensor([1.0, 2, 3, 4]))

  def test_mul_literal_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * 1)
  def test_mul_tensor_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * Tensor.ones(4))
  def test_literal_one_mul(self):
    _check_ast_count(0, 1 * Tensor([1.0, 2, 3, 4]))
  def test_tensor_one_mul(self):
    _check_ast_count(0, Tensor.ones(4) * Tensor([1.0, 2, 3, 4]))

  def test_div_literal_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) / 1)
  def test_div_tensor_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) / Tensor.ones(4))

  def test_pow_literal_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) ** 0)
  def test_pow_tensor_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) ** Tensor.zeros(4))

  def test_pow_literal_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) ** 1)
  def test_pow_tensor_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) ** Tensor.ones(4))
  # TODO: fix pow folding with left operand = 1
  @unittest.expectedFailure
  def test_literal_one_pow(self):
    _check_ast_count(0, 1 ** Tensor([1.0, 2, 3, 4]))
  @unittest.expectedFailure
  def test_tensor_one_pow(self):
    _check_ast_count(0, Tensor.ones(4) ** Tensor([1.0, 2, 3, 4]))

class TestMovedConstFolding(unittest.TestCase):
  def test_add_shrunk_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) + Tensor.zeros(6).shrink(((1, 5),)))

  def test_add_padded_zero(self):
    # TODO: it's 1 now, this might be possible to fold
    _check_ast_count(1, Tensor([1.0, 2, 3, 4]) + Tensor.zeros(2).pad(((1, 1),)))

  def test_mul_shrunk_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * Tensor.ones(6).shrink(((1, 5),)))

  def test_add_padded_one(self):
    _check_ast_count(1, Tensor([1.0, 2, 3, 4]) * Tensor.ones(2).pad(((1, 1),)))

@unittest.skipIf(CI and Device.DEFAULT in {"GPU", "CUDA", "METAL"}, "no GPU CI")
class TestMultiConstFolding(unittest.TestCase):
  def test_multi_const_folding_literal(self):
    ds = tuple(f"{Device.DEFAULT}:{i}" for i in range(4))
    t = Tensor.arange(16).float().realize().to(ds)

    # non const folding case creates one ast on each shard
    _check_ast_count(4, t + 1)
    _check_ast_count(4, 1 + t)
    _check_ast_count(4, t * 2)
    _check_ast_count(4, 2 * t)

    # const folded
    _check_ast_count(0, t + 0)
    _check_ast_count(0, 0 + t)
    _check_ast_count(0, t * 0)
    _check_ast_count(0, 0 * t)
    _check_ast_count(0, t * 1)
    _check_ast_count(0, 1 * t)
    np.testing.assert_equal((t + 0).numpy(), np.arange(16))
    np.testing.assert_equal((t * 0).numpy(), [0] * 16)
    np.testing.assert_equal((t * 1).numpy(), np.arange(16))

  @unittest.expectedFailure
  def test_multi_const_folding_tensor(self):
    ds = tuple(f"{Device.DEFAULT}:{i}" for i in range(4))
    t = Tensor.arange(16).float().realize().to(ds)
    zero = Tensor.zeros(16).realize().to(ds)
    one = Tensor.ones(16).realize().to(ds)

    # TODO: fix const to multi and const folding multi
    # const folded
    _check_ast_count(0, t + zero)
    _check_ast_count(0, zero + t)
    _check_ast_count(0, t * zero)
    _check_ast_count(0, zero * t)
    _check_ast_count(0, t * one)
    _check_ast_count(0, one * t)
    np.testing.assert_equal((t + zero).numpy(), np.arange(16))
    np.testing.assert_equal((t * zero).numpy(), [0] * 16)
    np.testing.assert_equal((t * one).numpy(), np.arange(16))

  @unittest.expectedFailure
  def test_multi_todo_pow(self):
    ds = tuple(f"{Device.DEFAULT}:{i}" for i in range(4))
    t = Tensor.arange(16).float().realize().to(ds)

    # TODO: fix pow folding
    _check_ast_count(0, t ** 0)
    _check_ast_count(0, t ** 1)
    _check_ast_count(0, 1 ** t)
