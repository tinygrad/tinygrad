import unittest
from tinygrad import Tensor
from tinygrad.ops import BufferOps
from tinygrad.engine.schedule import create_schedule

def _check_ast_count(desired_count:int, t:Tensor):
  # NOTE: this has side effect because everything can be scheduled only once
  asts = [s for s in create_schedule([t.lazydata]) if s.ast[0].op is BufferOps.STORE]
  assert len(asts) == desired_count

class TestSimpleConstFolding(unittest.TestCase):
  def test_add_literal_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) + 0)
  def test_add_tensor_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) + Tensor.zeros(4))

  def test_sub_literal_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) - 0)
  def test_sub_tensor_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) - Tensor.zeros(4))

  def test_mul_literal_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * 0)
  def test_mul_tensor_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * Tensor.zeros(4))

  def test_mul_literal_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * 1)
  def test_mul_tensor_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * Tensor.ones(4))

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
