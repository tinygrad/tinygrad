import unittest
import numpy as np
from tinygrad.helpers import (
  prod, dedup, argfix, argsort, all_same, all_int, colored, ansistrip, ansilen, make_tuple, flatten, fully_flatten,
  fromimport, strip_parens, ceildiv, round_up, round_down, next_power2, cdiv, cmod, lo32, hi32, data64, data64_le,
  getbits, i2u, is_numpy_ndarray, merge_dicts, partition, unwrap, get_single_element, get_child, word_wrap,
  pad_bytes, panic, polyN, pluralize, to_function_name, time_to_str, colorize_float, strides_for_shape,
  canonicalize_strides, get_contraction, Context, ContextVar, GlobalCounters, count,
)
from tinygrad.tensor import Tensor

class TestProd(unittest.TestCase):
  def test_basic(self): self.assertEqual(prod([1, 2, 3, 4]), 24)
  def test_empty(self): self.assertEqual(prod([]), 1)
  def test_single(self): self.assertEqual(prod([7]), 7)
  def test_with_zero(self): self.assertEqual(prod([5, 0, 3]), 0)
  def test_floats(self): self.assertAlmostEqual(prod([1.5, 2.0, 3.0]), 9.0)
  def test_tuple(self): self.assertEqual(prod((2, 3, 5)), 30)

class TestDedup(unittest.TestCase):
  def test_basic(self): self.assertEqual(dedup([1, 2, 2, 3, 1]), [1, 2, 3])
  def test_empty(self): self.assertEqual(dedup([]), [])
  def test_no_dupes(self): self.assertEqual(dedup([1, 2, 3]), [1, 2, 3])
  def test_all_same(self): self.assertEqual(dedup([5, 5, 5]), [5])
  def test_order_preserved(self): self.assertEqual(dedup([3, 1, 2, 1, 3]), [3, 1, 2])
  def test_strings(self): self.assertEqual(dedup(["a", "b", "a"]), ["a", "b"])

class TestArgfix(unittest.TestCase):
  def test_tuple_input(self): self.assertEqual(argfix((1, 2, 3)), (1, 2, 3))
  def test_list_input(self): self.assertEqual(argfix([1, 2, 3]), (1, 2, 3))
  def test_varargs(self): self.assertEqual(argfix(1, 2, 3), (1, 2, 3))
  def test_single(self): self.assertEqual(argfix(5), (5,))
  def test_bad_multiple_sequences(self):
    with self.assertRaises(ValueError): argfix((1,2), (3,4))

class TestArgsort(unittest.TestCase):
  def test_basic(self): self.assertEqual(argsort([3, 1, 2]), [1, 2, 0])
  def test_sorted(self): self.assertEqual(argsort([1, 2, 3]), [0, 1, 2])
  def test_reversed(self): self.assertEqual(argsort([3, 2, 1]), [2, 1, 0])
  def test_tuple_returns_tuple(self): self.assertIsInstance(argsort((3, 1, 2)), tuple)

class TestAllSame(unittest.TestCase):
  def test_same(self): self.assertTrue(all_same([1, 1, 1]))
  def test_different(self): self.assertFalse(all_same([1, 2, 1]))
  def test_empty(self): self.assertTrue(all_same([]))
  def test_single(self): self.assertTrue(all_same([42]))

class TestAllInt(unittest.TestCase):
  def test_all_ints(self): self.assertTrue(all_int((1, 2, 3)))
  def test_with_float(self): self.assertFalse(all_int((1, 2.0, 3)))
  def test_empty(self): self.assertTrue(all_int(()))
  def test_with_string(self): self.assertFalse(all_int((1, "2")))

class TestColored(unittest.TestCase):
  def test_none_color(self): self.assertEqual(colored("hi", None), "hi")
  def test_red(self): self.assertIn("\u001b[", colored("hi", "red"))
  def test_background(self): self.assertIn("\u001b[", colored("hi", "red", background=True))

class TestAnsistrip(unittest.TestCase):
  def test_no_ansi(self): self.assertEqual(ansistrip("hello"), "hello")
  def test_with_ansi(self): self.assertEqual(ansistrip(colored("hello", "red")), "hello")
  def test_empty(self): self.assertEqual(ansistrip(""), "")

class TestAnsilen(unittest.TestCase):
  def test_plain(self): self.assertEqual(ansilen("hello"), 5)
  def test_colored(self): self.assertEqual(ansilen(colored("hello", "red")), 5)

class TestMakeTuple(unittest.TestCase):
  def test_int_expand(self): self.assertEqual(make_tuple(3, 4), (3, 3, 3, 3))
  def test_sequence(self): self.assertEqual(make_tuple([1, 2, 3], 3), (1, 2, 3))
  def test_tuple_passthrough(self): self.assertEqual(make_tuple((1, 2), 2), (1, 2))

class TestFlatten(unittest.TestCase):
  def test_basic(self): self.assertEqual(flatten([[1, 2], [3, 4]]), [1, 2, 3, 4])
  def test_empty(self): self.assertEqual(flatten([]), [])
  def test_single(self): self.assertEqual(flatten([[1]]), [1])
  def test_mixed_lengths(self): self.assertEqual(flatten([[1], [2, 3], [4, 5, 6]]), [1, 2, 3, 4, 5, 6])

class TestFullyFlatten(unittest.TestCase):
  def test_nested(self): self.assertEqual(fully_flatten([[1, [2, 3]], [4]]), [1, 2, 3, 4])
  def test_string(self): self.assertEqual(fully_flatten("hello"), ["hello"])
  def test_scalar(self): self.assertEqual(fully_flatten(5), [5])
  def test_deep(self): self.assertEqual(fully_flatten([[[1]], [[2]]]), [1, 2])

class TestFromimport(unittest.TestCase):
  def test_import(self):
    result = fromimport("math", "sqrt")
    import math
    self.assertIs(result, math.sqrt)

class TestStripParens(unittest.TestCase):
  def test_basic(self): self.assertEqual(strip_parens("(abc)"), "abc")
  def test_no_parens(self): self.assertEqual(strip_parens("abc"), "abc")
  def test_nested(self): self.assertEqual(strip_parens("((a))"), "(a)")
  def test_unbalanced_inner(self): self.assertEqual(strip_parens("(a)(b)"), "(a)(b)")
  def test_empty_parens(self): self.assertEqual(strip_parens("()"), "")

class TestCeildiv(unittest.TestCase):
  def test_exact(self): self.assertEqual(ceildiv(10, 5), 2)
  def test_remainder(self): self.assertEqual(ceildiv(11, 5), 3)
  def test_one(self): self.assertEqual(ceildiv(1, 1), 1)
  def test_negative(self): self.assertEqual(ceildiv(-7, 2), -3)
  def test_large(self): self.assertEqual(ceildiv(1000, 3), 334)

class TestRoundUp(unittest.TestCase):
  def test_exact(self): self.assertEqual(round_up(16, 4), 16)
  def test_needs_rounding(self): self.assertEqual(round_up(17, 4), 20)
  def test_one(self): self.assertEqual(round_up(5, 1), 5)

class TestRoundDown(unittest.TestCase):
  def test_exact(self): self.assertEqual(round_down(16, 4), 16)
  def test_needs_rounding(self): self.assertEqual(round_down(17, 4), 16)
  def test_negative(self): self.assertEqual(round_down(-17, 4), -20)

class TestNextPower2(unittest.TestCase):
  def test_zero(self): self.assertEqual(next_power2(0), 1)
  def test_one(self): self.assertEqual(next_power2(1), 1)
  def test_power_of_two(self): self.assertEqual(next_power2(8), 8)
  def test_non_power(self): self.assertEqual(next_power2(5), 8)
  def test_large(self): self.assertEqual(next_power2(1000), 1024)

class TestCdiv(unittest.TestCase):
  def test_positive(self): self.assertEqual(cdiv(7, 2), 3)
  def test_negative_num(self): self.assertEqual(cdiv(-7, 2), -3)
  def test_negative_den(self): self.assertEqual(cdiv(7, -2), -3)
  def test_both_negative(self): self.assertEqual(cdiv(-7, -2), 3)
  def test_zero_den(self): self.assertEqual(cdiv(5, 0), 0)
  def test_exact(self): self.assertEqual(cdiv(6, 3), 2)

class TestCmod(unittest.TestCase):
  def test_positive(self): self.assertEqual(cmod(7, 2), 1)
  def test_negative(self): self.assertEqual(cmod(-7, 2), -1)
  def test_exact(self): self.assertEqual(cmod(6, 3), 0)

class TestBitOps(unittest.TestCase):
  def test_lo32(self): self.assertEqual(lo32(0x1_FFFFFFFF), 0xFFFFFFFF)
  def test_hi32(self): self.assertEqual(hi32(0x1_00000000), 1)
  def test_data64(self): self.assertEqual(data64(0x1_00000002), (1, 2))
  def test_data64_le(self): self.assertEqual(data64_le(0x1_00000002), (2, 1))
  def test_getbits(self): self.assertEqual(getbits(0b11010, 1, 3), 0b101)
  def test_getbits_single(self): self.assertEqual(getbits(0b1010, 3, 3), 1)

class TestI2u(unittest.TestCase):
  def test_positive(self): self.assertEqual(i2u(8, 5), 5)
  def test_negative(self): self.assertEqual(i2u(8, -1), 255)
  def test_negative_16(self): self.assertEqual(i2u(16, -1), 65535)

class TestIsNumpyNdarray(unittest.TestCase):
  def test_ndarray(self): self.assertTrue(is_numpy_ndarray(np.array([1, 2])))
  def test_list(self): self.assertFalse(is_numpy_ndarray([1, 2]))
  def test_tensor(self): self.assertFalse(is_numpy_ndarray(Tensor([1, 2])))

class TestMergeDicts(unittest.TestCase):
  def test_basic(self): self.assertEqual(merge_dicts([{"a": 1}, {"b": 2}]), {"a": 1, "b": 2})
  def test_same_key_same_value(self): self.assertEqual(merge_dicts([{"a": 1}, {"a": 1}]), {"a": 1})
  def test_conflict(self):
    with self.assertRaises(RuntimeError): merge_dicts([{"a": 1}, {"a": 2}])
  def test_empty(self): self.assertEqual(merge_dicts([]), {})

class TestPartition(unittest.TestCase):
  def test_basic(self):
    evens, odds = partition([1, 2, 3, 4, 5], lambda x: x % 2 == 0)
    self.assertEqual(evens, [2, 4])
    self.assertEqual(odds, [1, 3, 5])
  def test_all_true(self): self.assertEqual(partition([2, 4], lambda x: x % 2 == 0), ([2, 4], []))
  def test_empty(self): self.assertEqual(partition([], lambda x: True), ([], []))

class TestUnwrap(unittest.TestCase):
  def test_value(self): self.assertEqual(unwrap(5), 5)
  def test_none(self):
    with self.assertRaises(AssertionError): unwrap(None)

class TestGetSingleElement(unittest.TestCase):
  def test_single(self): self.assertEqual(get_single_element([42]), 42)
  def test_empty(self):
    with self.assertRaises(AssertionError): get_single_element([])
  def test_multiple(self):
    with self.assertRaises(AssertionError): get_single_element([1, 2])

class TestGetChild(unittest.TestCase):
  def test_dict(self): self.assertEqual(get_child({"a": {"b": 1}}, "a.b"), 1)
  def test_list_index(self): self.assertEqual(get_child([10, 20, 30], "1"), 20)
  def test_attr(self):
    class Obj:
      x = 5
    self.assertEqual(get_child(Obj(), "x"), 5)

class TestWordWrap(unittest.TestCase):
  def test_short(self): self.assertEqual(word_wrap("hello", wrap=80), "hello")
  def test_long(self):
    result = word_wrap("a" * 100, wrap=80)
    self.assertIn("\n", result)
  def test_multiline(self):
    result = word_wrap("short\n" + "a" * 100, wrap=80)
    lines = result.split("\n")
    self.assertTrue(len(lines) >= 3)

class TestPadBytes(unittest.TestCase):
  def test_aligned(self): self.assertEqual(pad_bytes(b"abcd", 4), b"abcd")
  def test_needs_padding(self): self.assertEqual(pad_bytes(b"abc", 4), b"abc\x00")
  def test_empty(self): self.assertEqual(pad_bytes(b"", 4), b"")
  def test_align_1(self): self.assertEqual(pad_bytes(b"abc", 1), b"abc")

class TestPanic(unittest.TestCase):
  def test_default(self):
    with self.assertRaises(RuntimeError): panic()
  def test_custom(self):
    with self.assertRaises(ValueError): panic(ValueError, "bad")

class TestPolyN(unittest.TestCase):
  def test_tensor(self):
    np.testing.assert_allclose(polyN(Tensor([1.0, 2.0, 3.0, 4.0]), [1.0, -2.0, 1.0]).numpy(), [0.0, 1.0, 4.0, 9.0])
  def test_float(self): self.assertAlmostEqual(polyN(2.0, [1.0, 0.0, 0.0]), 4.0)  # x^2
  def test_linear(self): self.assertAlmostEqual(polyN(3.0, [2.0, 1.0]), 7.0)  # 2x + 1
  def test_constant(self): self.assertAlmostEqual(polyN(99.0, [5.0]), 5.0)

class TestPluralize(unittest.TestCase):
  def test_singular(self): self.assertEqual(pluralize("item", 1), "1 item")
  def test_plural(self): self.assertEqual(pluralize("item", 3), "3 items")
  def test_zero(self): self.assertEqual(pluralize("item", 0), "0 items")

class TestToFunctionName(unittest.TestCase):
  def test_simple(self): self.assertEqual(to_function_name("hello"), "hello")
  def test_special_chars(self):
    result = to_function_name("a+b")
    self.assertNotIn("+", result)
    self.assertIn("a", result)
    self.assertIn("b", result)
  def test_with_ansi(self): self.assertEqual(to_function_name(colored("test", "red")), "test")

class TestTimeToStr(unittest.TestCase):
  def test_seconds(self): self.assertIn("s", time_to_str(15.0))
  def test_milliseconds(self): self.assertIn("ms", time_to_str(0.05))
  def test_microseconds(self): self.assertIn("us", time_to_str(0.000005))

class TestColorizeFloat(unittest.TestCase):
  def test_green(self): self.assertIn("green", repr(colorize_float(0.5).encode()))
  def test_red(self): self.assertIn("red", repr(colorize_float(2.0).encode()))

class TestStridesForShape(unittest.TestCase):
  def test_basic(self): self.assertEqual(strides_for_shape((2, 3, 4)), (12, 4, 1))
  def test_1d(self): self.assertEqual(strides_for_shape((5,)), (1,))
  def test_empty(self): self.assertEqual(strides_for_shape(()), ())
  def test_with_ones(self): self.assertEqual(strides_for_shape((1, 3, 1)), (3, 1, 1))

class TestCanonicalizeStrides(unittest.TestCase):
  def test_basic(self): self.assertEqual(canonicalize_strides((1, 3), (3, 1)), (0, 1))
  def test_no_ones(self): self.assertEqual(canonicalize_strides((2, 3), (3, 1)), (3, 1))

class TestGetContraction(unittest.TestCase):
  def test_identity(self): self.assertEqual(get_contraction((2, 3), (2, 3)), [[0], [1]])
  def test_merge(self): self.assertEqual(get_contraction((2, 3), (6,)), [[0, 1]])
  def test_impossible(self): self.assertIsNone(get_contraction((2, 3), (5,)))
  def test_add_ones(self): self.assertEqual(get_contraction((6,), (1, 6)), [[], [0]])

class TestContext(unittest.TestCase):
  def test_context_var(self):
    var = ContextVar("_test_var", 0)
    self.assertEqual(var.value, 0)
    with Context(**{"_test_var": 5}):
      self.assertEqual(var.value, 5)
    self.assertEqual(var.value, 0)

class TestGlobalCounters(unittest.TestCase):
  def test_reset(self):
    GlobalCounters.global_ops = 100
    GlobalCounters.reset()
    self.assertEqual(GlobalCounters.global_ops, 0)
    self.assertEqual(GlobalCounters.global_mem, 0)

class TestCount(unittest.TestCase):
  def test_increment(self):
    c = count(0)
    self.assertEqual(c(), 0)
    self.assertEqual(c(), 1)
    self.assertEqual(c(), 2)
  def test_custom_start(self):
    c = count(10)
    self.assertEqual(c(), 10)
    self.assertEqual(c(), 11)

if __name__ == '__main__':
  unittest.main()
