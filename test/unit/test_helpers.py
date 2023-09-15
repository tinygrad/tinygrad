import unittest
import numpy as np
from tinygrad.helpers import Context, ContextVar, DType, dtypes, merge_dicts, strip_parens, prod
from tinygrad.shape.symbolic import Variable, NumNode

VARIABLE = ContextVar("VARIABLE", 0)

class TestContextVars(unittest.TestCase):
  # Ensuring that the test does not modify variables outside the tests.
  ctx = Context()
  def setUp(self): TestContextVars.ctx.__enter__()
  def tearDown(self): TestContextVars.ctx.__exit__()

  def test_initial_value_is_set(self):
    _TMP = ContextVar("_TMP", 5)
    self.assertEqual(_TMP.value, 5)

  def test_multiple_creation_ignored(self):
    _TMP2 = ContextVar("_TMP2", 1)
    _TMP2 = ContextVar("_TMP2", 2)
    self.assertEqual(_TMP2.value, 1)

  def test_new_var_inside_context(self):
    # Creating a _new_ variable inside a context should not have any effect on its scope (?)
    with Context(VARIABLE=1):
      _TMP3 = ContextVar("_TMP3", 1)
    _TMP3 = ContextVar("_TMP3", 2)
    self.assertEqual(_TMP3.value, 1)

  def test_value_accross_modules(self):
    # Mocking module import by invoking the code but not in our globals().
    exec('from tinygrad.helpers import ContextVar;C = ContextVar("C", 13)', {}) # pylint:disable=exec-used
    # It should not matter that the first creation was in another module.
    C = ContextVar("C", 0)
    self.assertEqual(C.value, 13)

  def test_assignment_across_modules(self):
    B = ContextVar("B", 1)
    # local assignment
    B.value = 2
    self.assertEqual(B.value, 2)
    # Assignment in another module.
    exec('from tinygrad.helpers import ContextVar;B = ContextVar("B", 0);B.value = 3;', {}) # pylint:disable=exec-used
    # Assignment in another module should affect this one as well.
    self.assertEqual(B.value, 3)

  def test_context_assignment(self):
    with Context(VARIABLE=1):
      self.assertEqual(VARIABLE.value, 1)
    self.assertEqual(VARIABLE.value, 0)

  def test_unknown_param_to_context(self):
    with self.assertRaises(KeyError):
      with Context(SOMETHING_ELSE=1):
        pass

  def test_inside_context_assignment(self):
    with Context(VARIABLE=4):
      # What you can and cannot do inside a context.
      # 1. This type of statement has no effect.
      VARIABLE = ContextVar("VARIABLE", 0)
      self.assertTrue(VARIABLE >= 4, "ContextVars inside contextmanager may not set a new value")

      # 2. The call syntax however has a local effect.
      VARIABLE.value = 13
      self.assertTrue(VARIABLE.value == 13, "Call syntax however works inside a contextmanager.")

    # Related to 2. above. Note that VARIABLE is back to 0 again as expected.
    self.assertEqual(VARIABLE.value, 0)

  def test_new_var_inside_context_other_module(self):
    with Context(VARIABLE=1):
      _NEW2 = ContextVar("_NEW2", 0)
    _NEW2 = ContextVar("_NEW2", 1)
    self.assertEqual(_NEW2.value, 0)

    code = """\
from tinygrad.helpers import Context, ContextVar
with Context(VARIABLE=1):
  _NEW3 = ContextVar("_NEW3", 0)"""
    exec(code, {})  # pylint:disable=exec-used
    # While _NEW3 was created in an outside scope it should still work the same as above.
    _NEW3 = ContextVar("_NEW3", 1)
    self.assertEqual(_NEW3.value, 0)

  def test_nested_context(self):
    with Context(VARIABLE=1):
      with Context(VARIABLE=2):
        with Context(VARIABLE=3):
          self.assertEqual(VARIABLE.value, 3)
        self.assertEqual(VARIABLE.value, 2)
      self.assertEqual(VARIABLE.value, 1)
    self.assertEqual(VARIABLE.value, 0)

  def test_decorator(self):
    @Context(VARIABLE=1, DEBUG=4)
    def test():
      self.assertEqual(VARIABLE.value, 1)

    self.assertEqual(VARIABLE.value, 0)
    test()
    self.assertEqual(VARIABLE.value, 0)

  def test_context_exit_reverts_updated_values(self):
    D = ContextVar("D", 1)
    D.value = 2
    with Context(D=3):
      ...
    assert D.value == 2, f"Expected D to be 2, but was {D.value}. Indicates that Context.__exit__ did not restore to the correct value."

class TestMergeDicts(unittest.TestCase):
  def test_merge_dicts(self):
    a = {"a": 1, "b": 2}
    b = {"a": 1, "c": 3}
    c = {}
    d = {"a": 2, "b": 2}
    assert merge_dicts([a, b]) == {"a": 1, "b": 2, "c": 3}
    assert merge_dicts([a, c]) == a
    assert merge_dicts([a, b, c]) == {"a": 1, "b": 2, "c": 3}
    with self.assertRaises(AssertionError):
      merge_dicts([a, d])

class TestDtypes(unittest.TestCase):
  def test_dtypes_fields(self):
    fields = dtypes.fields()
    self.assertTrue(all(isinstance(value, DType) for value in fields.values()))
    self.assertTrue(all(issubclass(value.np, np.generic) for value in fields.values() if value.np is not None))

class TestStripParens(unittest.TestCase):
  def test_simple(self): self.assertEqual("1+2", strip_parens("(1+2)"))
  def test_nested(self): self.assertEqual("1+(2+3)", strip_parens("(1+(2+3))"))
  def test_casted_no_strip(self): self.assertEqual("(int)(1+2)", strip_parens("(int)(1+2)"))

class TestProd(unittest.TestCase):
  def test_empty(self): self.assertEqual(1, prod(tuple()))
  def test_ints(self): self.assertEqual(30, prod((2, 3, 5)))
  def test_variable(self): self.assertEqual("(a*12)", prod((Variable("a", 1, 5), 3, 4)).render())
  def test_variable_order(self): self.assertEqual("(a*12)", prod((3, 4, Variable("a", 1, 5))).render())
  def test_num_nodes(self): self.assertEqual(NumNode(6), prod((NumNode(2), NumNode(3))))

if __name__ == '__main__':
  unittest.main()