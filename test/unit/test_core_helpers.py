import unittest
from tinygrad.helpers import Context, ContextVar

# TODO unsure about the name and placement of this new file.
# avoided naming it test_helpers.py since there already is already extras/test_helpers.py

VARIABLE = ContextVar("VARIABLE", 0)

class TestContextVars(unittest.TestCase):
  def test_initial_value(self):
    self.assertEqual(VARIABLE.value, 0)

  def test_assignment_by_functioncall(self):
    global VARIABLE
    VARIABLE(6)
    self.assertEqual(VARIABLE.value, 6)

  def test_context_value(self):
    global VARIABLE
    VARIABLE(0)
    self.assertEqual(VARIABLE.value, 0)

    with Context(VARIABLE=4):
      self.assertEqual(VARIABLE.value, 4)

      VARIABLE = ContextVar("VARIABLE", 0)
      self.assertTrue(VARIABLE >= 4, "ContextVars inside contextmanager shall not override.")

      # Not sure if Context in Context is strictly necessary.
      with Context(VARIABLE=6):
        self.assertEqual(VARIABLE.value, 6)
      self.assertEqual(VARIABLE.value, 4)

    self.assertEqual(VARIABLE.value, 0)

if __name__ == '__main__':
  unittest.main()