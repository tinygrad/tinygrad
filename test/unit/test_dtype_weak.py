import unittest
from unittest.mock import patch

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp


class TestWeakPromotion(unittest.TestCase):
  def test_uop_scalar_const_unchanged(self):
    for dtype, value in ((dtypes.index, 1), (dtypes.int32, 1), (dtypes.float32, 0.5)):
      out = UOp.variable("x", 0.0 if dtype == dtypes.float32 else 0, 10.0 if dtype == dtypes.float32 else 10, dtype) + value
      self.assertEqual((out.dtype, out.src[1].dtype), (dtype, dtype))

  @unittest.expectedFailure  # TODO: a weak const defers to its consumer (JAX): these dtypes change once python scalars are weak consts
  def test_changed_rows(self):
    t_i8, t_f16, t_bf16 = Tensor([1], dtype=dtypes.int8), Tensor([1], dtype=dtypes.float16), Tensor([1], dtype=dtypes.bfloat16)
    t_bool, t_u16 = Tensor([True]), Tensor([1], dtype=dtypes.uint16)
    self.assertEqual((t_i8 + 0.5).dtype, dtypes.weakfloat)
    self.assertEqual(((t_i8 + 0.5) + t_f16).dtype, dtypes.float16)
    self.assertEqual(((t_i8 + 0.5) + t_bf16).dtype, dtypes.bfloat16)
    self.assertEqual(((t_bool + 1) + t_i8).dtype, dtypes.int8)
    self.assertEqual(((t_bool + 1) + t_u16).dtype, dtypes.uint16)
    self.assertEqual((Tensor(3) + t_i8).dtype, dtypes.int8)
    self.assertEqual(Tensor([2], dtype=dtypes.uint8).pad(((1, 1),), value=1).dtype, dtypes.uint8)

  def test_unchanged_rows(self):
    t_i8, t_f16, t_f32 = Tensor([1], dtype=dtypes.int8), Tensor([1], dtype=dtypes.float16), Tensor([1], dtype=dtypes.float32)
    self.assertEqual((t_i8 + 1).dtype, dtypes.int8)
    self.assertEqual((t_f16 + 0.5).dtype, dtypes.float16)
    self.assertEqual((t_f32 + t_f16).dtype, dtypes.float32)

  @unittest.expectedFailure  # TODO: dot of a weak const tensor defers to the other operand once python scalars are weak consts
  def test_dot_defers_weak(self):
    weak = Tensor([True, False]).where(Tensor(1), 2)
    self.assertEqual(weak.dot(Tensor([1, 1], dtype=dtypes.int8)).dtype, dtypes.int8)

  @unittest.expectedFailure  # TODO: Tensor(3).uop becomes CONST(weakint); Tensor.dtype is always uop.dtype; buffers lower to the default
  def test_dtype_is_uop_dtype(self):
    for value, weak, lowered in ((3, dtypes.weakint, dtypes.default_int), (0.5, dtypes.weakfloat, dtypes.default_float)):
      t = Tensor(value)
      self.assertEqual((t.uop.dtype, t.dtype), (weak, weak))
      self.assertEqual(t.numpy().dtype.itemsize, lowered.itemsize)
      realized = t.clone("CPU").realize()
      self.assertEqual((realized.dtype, realized.uop.buffer.dtype), (lowered, lowered))
    with patch.object(dtypes, "default_int", dtypes.int64):
      self.assertEqual(Tensor(3).clone("CPU").realize().uop.buffer.dtype, dtypes.int64)

  def test_integer_values(self):
    x = Tensor.full((1,), 1, dtype=dtypes.int64, device="CPU")
    self.assertEqual((x + 2**40).item(), 2**40 + 1)
    self.assertEqual((x << 3).item(), 8)
    self.assertTrue((x < 2**40).item())

  def test_float64_precision(self):
    value = 1.0 + 2**-40
    x64 = Tensor.full((1,), 1.0, dtype=dtypes.float64, device="CPU")
    self.assertEqual((x64 + value).item(), 2.0 + 2**-40)
    x32 = Tensor.full((1,), 0.0, dtype=dtypes.float32, device="CPU")
    self.assertEqual((x32 + value).item(), 1.0)

  @unittest.expectedFailure  # TODO: exp/cos/sigmoid of a weak const stay weak instead of casting to a concrete float
  def test_weak_transcendentals(self):
    t_f16 = Tensor([1], dtype=dtypes.float16)
    for out in (Tensor(2).exp(), Tensor(2).cos(), Tensor(2).sigmoid()):
      self.assertEqual((out.dtype, (out + t_f16).dtype), (dtypes.weakfloat, dtypes.float16))

  @unittest.expectedFailure  # TODO: where of weak consts stays weak and resolves per consumer
  def test_where_and_shared_literal(self):
    gate, weak = Tensor([True, False], device="CPU"), Tensor(2)
    weak_where = gate.where(weak, 3)
    self.assertEqual(weak_where.dtype, dtypes.weakint)
    self.assertEqual((weak_where + Tensor([1, 1], dtype=dtypes.int64, device="CPU")).tolist(), [3, 4])
    self.assertEqual((weak + Tensor([1], dtype=dtypes.int32, device="CPU")).item(), 3)
    self.assertEqual((weak + Tensor([1], dtype=dtypes.int64, device="CPU")).item(), 3)

  @unittest.expectedFailure  # TODO: weak consts lower to concrete dtypes before kernels
  def test_null_lowering(self):
    for t in (Tensor.full((1,), 1, dtype=dtypes.int64, device="NULL") + 2**40,
              Tensor.full((1,), 1.0, dtype=dtypes.float64, device="NULL") + (1.0 + 2**-40)):
      t.realize()
      self.assertNotIn(t.uop.buffer.dtype, dtypes.weaks)


if __name__ == "__main__":
  unittest.main()
