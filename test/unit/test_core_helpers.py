import unittest
from tinygrad.helpers import Context, ContextVar

# TODO unsure about the name and placement of this new file.
# avoided naming it test_helpers.py since there already is already extras/test_helpers.py

VARIABLE = ContextVar("VARIABLE", 0)

class TestContextVars(unittest.TestCase):
  # Ensuring that the test does not modify variables outside the tests.
  ctx = Context()
  def setUp(self): TestContextVars.ctx.__enter__()
  def tearDown(self): TestContextVars.ctx.__exit__()

  def test_initial_value_is_set(self):
    _TMP = ContextVar("_TMP", 5)
    self.assertEqual(_TMP.value, 5)

  def test_assignment_by_call_syntax(self):
    VARIABLE(6)
    self.assertEqual(VARIABLE.value, 6)

  def test_context_assignment(self):
    with Context(VARIABLE=1):
      self.assertEqual(VARIABLE.value, 1)
    self.assertEqual(VARIABLE.value, 0)

  def test_inside_context_assignment(self):
    with Context(VARIABLE=4):
      # What you can and cannot do inside a context.
      # 1. This type of statement has no effect.
      VARIABLE = ContextVar("VARIABLE", 0)
      self.assertTrue(VARIABLE >= 4, "ContextVars inside contextmanager may not set a new value")

      # 2. The call syntax however has a local effect.
      VARIABLE(13)
      self.assertTrue(VARIABLE.value == 13, "Call syntax however works inside a contextmanager.")

    # Related to 2. above. Note that VARIABLE is back to 0 again as expected.
    self.assertEqual(VARIABLE.value, 0)

  def test_new_var_inside_context(self):
    with Context(VARIABLE=1):
      _NEW = ContextVar("_NEW", 0)
    self.assertEqual(_NEW.value, 0)

  def test_nested_context(self):
    with Context(VARIABLE=1):
      with Context(VARIABLE=2):
        self.assertEqual(VARIABLE.value, 2)
      self.assertEqual(VARIABLE.value, 1)
    self.assertEqual(VARIABLE.value, 0)

if __name__ == '__main__':
  unittest.main()