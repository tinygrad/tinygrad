#!/usr/bin/env python
"""
JIT Footguns: Documenting unexpected behavior changes when using @TinyJit

Each test demonstrates a pattern that works without JIT but behaves differently with JIT.
Tests document both the ACTUAL behavior and the EXPECTED behavior (what users would intuitively expect).
"""
import unittest
import numpy as np
from tinygrad import Tensor, TinyJit

class TestOutputBufferReuse(unittest.TestCase):
  """
  FOOTGUN: JIT reuses output buffers, so holding references to old outputs gives stale data.

  Without JIT: Each call returns a new tensor with independent data.
  With JIT: Output tensors share the same underlying buffer, so old references see new values.

  EXPECTED BEHAVIOR: JIT should behave like non-JIT - each output should be independent,
  OR the documentation should clearly warn that outputs are invalidated on next call.
  """

  def test_without_jit(self):
    def f(x): return x.sum().realize()

    result_1 = f(Tensor([1, 1]))
    result_2 = f(Tensor([2, 2]))
    result_3 = f(Tensor([3, 3]))

    # without JIT, each result is independent
    self.assertEqual(result_1.item(), 2)
    self.assertEqual(result_2.item(), 4)
    self.assertEqual(result_3.item(), 6)

  def test_with_jit(self):
    @TinyJit
    def f(x): return x.sum().realize()

    result_1 = f(Tensor([1, 1]))  # call 0: warmup
    result_2 = f(Tensor([2, 2]))  # call 1: capture
    result_3 = f(Tensor([3, 3]))  # call 2: jit exec

    # only result_1 (warmup) is independent
    self.assertEqual(result_1.item(), 2)

    # ACTUAL: result_2 (capture) and result_3 share the same buffer
    # result_3 is correct (latest)
    self.assertEqual(result_3.item(), 6)
    # result_2 is overwritten by result_3 - should be 4!
    self.assertEqual(result_2.item(), 6)

  def test_workaround_clone(self):
    """Use .clone().realize() to get independent copies"""
    @TinyJit
    def f(x): return x.sum().realize()

    result_1 = f(Tensor([1, 1])).clone().realize()
    result_2 = f(Tensor([2, 2])).clone().realize()
    result_3 = f(Tensor([3, 3])).clone().realize()

    self.assertEqual(result_1.item(), 2)
    self.assertEqual(result_2.item(), 4)
    self.assertEqual(result_3.item(), 6)


class TestNonTensorOutputs(unittest.TestCase):
  """
  FOOTGUN: Non-tensor return values are captured and frozen at JIT capture time.

  Without JIT: Non-tensor values are computed fresh each call.
  With JIT: Non-tensor values are frozen to whatever they were during capture.

  EXPECTED BEHAVIOR: Either non-tensor outputs should work correctly,
  OR attempting to return non-tensor values should raise an error.
  """

  def test_without_jit(self):
    def f(x, multiplier):
      return (x * 2).realize(), multiplier * 10

    for i in range(5):
      tensor_out, scalar_out = f(Tensor([i]), i)
      self.assertEqual(tensor_out.item(), i * 2)
      self.assertEqual(scalar_out, i * 10)

  def test_with_jit(self):
    @TinyJit
    def f(x, multiplier):
      return (x * 2).realize(), multiplier * 10

    results = []
    for i in range(5):
      tensor_out, scalar_out = f(Tensor([i]), i)
      results.append((tensor_out.item(), scalar_out))

    # tensor outputs work correctly
    self.assertEqual(results[2][0], 4)  # 2 * 2
    self.assertEqual(results[3][0], 6)  # 3 * 2
    self.assertEqual(results[4][0], 8)  # 4 * 2

    # ACTUAL: scalar outputs are frozen at capture time (i=1)
    # After warmup (i=0,1), capture happens at i=1, so scalar_out = 10 forever
    self.assertEqual(results[2][1], 10)  # should be 20!
    self.assertEqual(results[3][1], 10)  # should be 30!
    self.assertEqual(results[4][1], 10)  # should be 40!


class TestDuplicateInputs(unittest.TestCase):
  """
  FOOTGUN: JIT cannot handle the same tensor passed as multiple arguments.

  Without JIT: Works fine - operations see the same data.
  With JIT: Raises AssertionError about duplicate inputs.

  EXPECTED BEHAVIOR: Should work or provide a clear error message explaining why.
  """

  def test_without_jit(self):
    def f(a, b): return (a + b).realize()

    x = Tensor([1, 2, 3])
    result = f(x, x)
    np.testing.assert_array_equal(result.numpy(), [2, 4, 6])

  def test_with_jit(self):
    @TinyJit
    def f(a, b): return (a + b).realize()

    x = Tensor([1, 2, 3])
    # ACTUAL: Raises AssertionError: duplicate inputs to JIT
    with self.assertRaises(AssertionError):
      f(x, x)


class TestTensorsInContainers(unittest.TestCase):
  """
  FOOTGUN: Tensors inside lists/dicts are not tracked as inputs.

  Without JIT: Tensors in containers are used correctly.
  With JIT: Only the tensor from capture time is used; new tensors are ignored.

  EXPECTED BEHAVIOR: Tensors in containers should be properly tracked,
  OR there should be a clear error/warning.
  """

  def test_without_jit(self):
    def f(a, arr): return (a + arr[0]).realize()

    for i in range(5):
      a = Tensor([1, 1, 1])
      b = Tensor([i, i, i])
      result = f(a, [b])
      np.testing.assert_array_equal(result.numpy(), [1 + i, 1 + i, 1 + i])

  def test_with_jit(self):
    @TinyJit
    def f(a, arr): return (a + arr[0]).realize()

    results = []
    for i in range(5):
      a = Tensor([1, 1, 1]).realize()
      b = Tensor([i, i, i]).realize()
      result = f(a, [b])
      results.append(result.numpy().copy())

    # first two calls are warmup/capture, work correctly
    np.testing.assert_array_equal(results[0], [1, 1, 1])  # 1 + 0
    np.testing.assert_array_equal(results[1], [2, 2, 2])  # 1 + 1

    # ACTUAL: after capture, arr[0] is frozen to the tensor from i=1
    # so results[2] should be [3,3,3] but is still using the old b
    np.testing.assert_array_equal(results[2], [2, 2, 2])  # should be [3,3,3]!


class TestNestedJit(unittest.TestCase):
  """
  FOOTGUN: Nested JIT (JIT inside JIT) fails, but only on the second call.

  Without JIT: Nested calls work fine.
  With JIT: First call works, second call raises RuntimeError.

  EXPECTED BEHAVIOR: Should either work correctly or fail consistently on first call.
  """

  def test_without_jit(self):
    def inner(t): return t + 1
    def outer(t): return inner(t) * 3

    for i in range(5):
      result = outer(Tensor([i]))
      self.assertEqual(result.item(), (i + 1) * 3)

  def test_with_jit(self):
    @TinyJit
    def inner(t): return t + 1

    @TinyJit
    def outer(t): return inner(t) * 3

    # first call works (confusingly!)
    result = outer(Tensor([1])).realize()
    self.assertEqual(result.item(), 6)

    # ACTUAL: second call fails
    with self.assertRaises(RuntimeError):
      outer(Tensor([2])).realize()


class TestImplicitInputsNeedRealize(unittest.TestCase):
  """
  FOOTGUN: Implicit inputs (closures) must be .realize()'d before the JIT call.

  Without JIT: Assignment propagates automatically.
  With JIT: Without explicit realize, the old value is used.

  EXPECTED BEHAVIOR: Should work without explicit realize, or documentation
  should clearly explain this requirement.
  """

  def test_without_jit(self):
    x = Tensor([0])

    def f():
      return (x * 2).realize()

    for i in range(5):
      x.assign(Tensor([i]))  # no realize!
      result = f()
      self.assertEqual(result.item(), i * 2)

  def test_with_jit_broken(self):
    x = Tensor([0])

    @TinyJit
    def f():
      return (x * 2).realize()

    results = []
    for i in range(5):
      x.assign(Tensor([i]))  # no realize - this is the bug
      result = f()
      results.append(result.item())

    # ACTUAL: without realize, x doesn't update properly
    # The exact behavior depends on timing, but it won't be correct
    # This test documents that the pattern doesn't work reliably

  def test_with_jit_correct(self):
    x = Tensor([0])

    @TinyJit
    def f():
      return (x * 2).realize()

    for i in range(5):
      x.assign(Tensor([i])).realize()  # must realize!
      result = f()
      self.assertEqual(result.item(), i * 2)


class TestViewsWithDifferentOffsets(unittest.TestCase):
  """
  FOOTGUN: JIT requires consistent tensor views/shapes across calls.

  Without JIT: Views with different offsets work fine.
  With JIT: Different offsets cause assertion errors.

  EXPECTED BEHAVIOR: Either handle variable offsets or provide clearer error.
  """

  def test_without_jit(self):
    def f(a): return (a + 1).realize()

    base = Tensor.randn(10, 10).realize()
    for i in range(1, 5):
      a = base[:, i:i+2]  # different offset each time
      result = f(a)
      np.testing.assert_allclose(result.numpy(), a.numpy() + 1, atol=1e-5)

  def test_with_jit(self):
    @TinyJit
    def f(a): return (a + 1).realize()

    base = Tensor.randn(10, 10).realize()
    # ACTUAL: raises AssertionError because view offset changes
    with self.assertRaises(AssertionError):
      for i in range(1, 5):
        a = base[:, i:i+2]
        f(a)


class TestShapeChangeAfterCapture(unittest.TestCase):
  """
  FOOTGUN: JIT requires consistent shapes, but errors can be confusing.

  Without JIT: Different shapes work (assuming operation supports them).
  With JIT: Shape mismatch causes assertion error, but only after capture phase.

  EXPECTED BEHAVIOR: Clearer error message explaining the shape constraint.
  """

  def test_without_jit(self):
    def f(a, b): return (a + b).realize()

    # different shapes work fine
    f(Tensor.randn(10, 10), Tensor.randn(10, 10))
    f(Tensor.randn(20, 20), Tensor.randn(20, 20))

  def test_with_jit(self):
    @TinyJit
    def f(a, b): return (a + b).realize()

    # warmup and capture with shape (10, 10)
    f(Tensor.randn(10, 10), Tensor.randn(10, 10))
    f(Tensor.randn(10, 10), Tensor.randn(10, 10))

    # ACTUAL: different shape fails with assertion error
    with self.assertRaises(AssertionError):
      f(Tensor.randn(20, 20), Tensor.randn(20, 20))


class TestConstantsInsideJit(unittest.TestCase):
  """
  FOOTGUN: Python constants/variables inside JIT are captured at definition time.

  Without JIT: Uses current value of Python variable.
  With JIT: Uses value at capture time.

  EXPECTED BEHAVIOR: Either track Python variables or document this clearly.
  """

  def test_without_jit(self):
    multiplier = 1

    def f(x):
      return (x * multiplier).realize()

    results = []
    for i in range(5):
      multiplier = i + 1
      result = f(Tensor([10]))
      results.append(result.item())

    self.assertEqual(results, [10, 20, 30, 40, 50])

  def test_with_jit(self):
    multiplier = 1

    @TinyJit
    def f(x):
      return (x * multiplier).realize()

    results = []
    for i in range(5):
      multiplier = i + 1
      result = f(Tensor([10]))
      results.append(result.item())

    # ACTUAL: multiplier is frozen at capture time (i=1, so multiplier=2)
    # First call (i=0): warmup, multiplier=1, returns 10
    # Second call (i=1): capture, multiplier=2, returns 20
    # After capture: multiplier frozen at 2
    self.assertEqual(results[0], 10)   # warmup
    self.assertEqual(results[1], 20)   # capture
    self.assertEqual(results[2], 20)   # should be 30!
    self.assertEqual(results[3], 20)   # should be 40!
    self.assertEqual(results[4], 20)   # should be 50!


class TestConditionalLogic(unittest.TestCase):
  """
  FOOTGUN: Conditional control flow is captured at JIT time, not evaluated dynamically.

  Without JIT: Condition is evaluated each call.
  With JIT: The branch taken during capture is always used.

  EXPECTED BEHAVIOR: Either trace both branches or document this clearly.
  """

  def test_without_jit(self):
    def f(x, use_square):
      if use_square:
        return (x * x).realize()
      else:
        return (x * 2).realize()

    self.assertEqual(f(Tensor([3]), True).item(), 9)
    self.assertEqual(f(Tensor([3]), False).item(), 6)
    self.assertEqual(f(Tensor([3]), True).item(), 9)

  def test_with_jit(self):
    @TinyJit
    def f(x, use_square):
      if use_square:
        return (x * x).realize()
      else:
        return (x * 2).realize()

    # warmup with use_square=True
    f(Tensor([3]), True)
    # capture with use_square=False
    f(Tensor([3]), False)

    # ACTUAL: now we're stuck with the False branch forever
    result = f(Tensor([3]), True)  # passing True but False branch runs
    self.assertEqual(result.item(), 6)  # should be 9!


class TestRandomInJit(unittest.TestCase):
  """
  FOOTGUN: Random tensors inside JIT regenerate each call (which is good!),
  but the behavior with unrealized random tensors before JIT can be confusing.

  This is actually handled correctly, documenting expected behavior.
  """

  def test_random_regenerates(self):
    @TinyJit
    def f(x):
      rand = Tensor.rand(3)
      return (x + rand).realize()

    # warmup and capture
    f(Tensor([0, 0, 0]))
    f(Tensor([0, 0, 0]))

    # random should be different each call
    results = set()
    for _ in range(5):
      result = f(Tensor([0, 0, 0]))
      results.add(tuple(result.numpy().tolist()))

    # all 5 results should be different
    self.assertEqual(len(results), 5, "Random should regenerate in JIT")


class TestNothingRealized(unittest.TestCase):
  """
  FOOTGUN: If no kernels run during JIT capture, it raises an assertion error.

  Without JIT: Returning unrealized tensors is fine.
  With JIT: Must realize something or JIT fails.

  EXPECTED BEHAVIOR: Clearer error message or allow empty JIT.
  """

  def test_without_jit(self):
    def f(a, b):
      return None  # returns nothing, does nothing

    for _ in range(5):
      result = f(Tensor([1]), Tensor([2]))
      self.assertIsNone(result)

  def test_with_jit(self):
    @TinyJit
    def f(a, b):
      return None

    # ACTUAL: raises AssertionError "didn't JIT anything!"
    with self.assertRaises(AssertionError):
      for _ in range(5):
        f(Tensor([1]), Tensor([2]))


class TestReturnUnrealizedTensor(unittest.TestCase):
  """
  NOTE: This actually works correctly - unrealized return tensors are auto-realized.
  Documenting expected behavior.
  """

  def test_unrealized_return_works(self):
    @TinyJit
    def f(a, b):
      return a + b  # no explicit realize

    for _ in range(5):
      a = Tensor.randn(10)
      b = Tensor.randn(10)
      result = f(a, b)
      np.testing.assert_allclose(result.numpy(), a.numpy() + b.numpy(), atol=1e-5)


class TestEmptyTensorSizeMismatch(unittest.TestCase):
  """
  FOOTGUN: Empty tensors have uninitialized data, leading to confusing results
  when mixed with real tensors.

  Without JIT: Each call processes the actual tensor.
  With JIT: Empty tensor during warmup/capture means kernels read garbage.

  EXPECTED BEHAVIOR: Clearer documentation about empty tensors.
  """

  def test_without_jit(self):
    def f(x): return (x + 1).realize()

    # works with any size
    f(Tensor.empty(1))
    f(Tensor.empty(10))
    f(Tensor([2.0]))
    self.assertEqual(f(Tensor([5.0])).item(), 6.0)

  def test_with_jit_shape_mismatch(self):
    @TinyJit
    def f(x): return (x + 1).realize()

    # warmup/capture with shape (1,)
    f(Tensor([1.0]))
    f(Tensor([1.0]))

    # scalar tensor has shape () - different from (1,)
    # ACTUAL: raises assertion error for shape mismatch
    with self.assertRaises(AssertionError):
      f(Tensor(2.0))


class TestKwargsOrderMatters(unittest.TestCase):
  """
  NOTE: This actually works correctly - kwargs are sorted by name.
  Documenting expected behavior.
  """

  def test_kwargs_order_consistent(self):
    @TinyJit
    def f(first, second): return (first / second).realize()

    # order in call doesn't matter - sorted by kwarg name
    for _ in range(3):
      a = Tensor.randn(10)
      b = Tensor.randn(10) + 1  # avoid div by zero
      result = f(second=b, first=a)
      np.testing.assert_allclose(result.numpy(), a.numpy() / b.numpy(), atol=1e-4)

    for _ in range(3):
      a = Tensor.randn(10)
      b = Tensor.randn(10) + 1
      result = f(first=a, second=b)
      np.testing.assert_allclose(result.numpy(), a.numpy() / b.numpy(), atol=1e-4)


class TestMixingPositionalAndKwargs(unittest.TestCase):
  """
  FOOTGUN: Mixing positional and keyword args is NOT allowed after capture.

  Without JIT: Positional and kwargs can be used interchangeably.
  With JIT: Must use same calling convention after capture.

  EXPECTED BEHAVIOR: Either support both, or provide clearer error message.
  """

  def test_without_jit(self):
    def f(a, b): return (a + b).realize()

    # positional works
    self.assertEqual(f(Tensor([1]), Tensor([2])).item(), 3)
    # kwargs work
    self.assertEqual(f(a=Tensor([3]), b=Tensor([4])).item(), 7)
    # can switch freely
    self.assertEqual(f(Tensor([5]), Tensor([6])).item(), 11)

  def test_with_jit(self):
    @TinyJit
    def f(a, b): return (a + b).realize()

    # warmup and capture with positional
    f(Tensor([1]), Tensor([2]))
    f(Tensor([1]), Tensor([2]))

    # ACTUAL: switching to kwargs fails with "args mismatch"
    # expected_names=[0, 1] != ['a', 'b']
    with self.assertRaises(AssertionError):
      f(a=Tensor([3]), b=Tensor([4]))


class TestClassMethodJit(unittest.TestCase):
  """
  FOOTGUN: JIT on instance methods can lead to shared state across instances.

  Without JIT: Each instance operates independently.
  With JIT: The JIT is shared if decorator is on the class.

  EXPECTED BEHAVIOR: Should work per-instance or clearly document the sharing.
  """

  def test_without_jit(self):
    class Model:
      def __init__(self, scale):
        self.scale = Tensor([scale])

      def forward(self, x):
        return (x * self.scale).realize()

    m1 = Model(2)
    m2 = Model(3)

    self.assertEqual(m1.forward(Tensor([5])).item(), 10)
    self.assertEqual(m2.forward(Tensor([5])).item(), 15)

  def test_with_jit(self):
    class Model:
      def __init__(self, scale):
        self.scale = Tensor([scale])

      @TinyJit
      def forward(self, x):
        return (x * self.scale).realize()

    m1 = Model(2)
    m2 = Model(3)

    # warmup and capture on m1
    m1.forward(Tensor([5]))
    m1.forward(Tensor([5]))
    # JIT is now captured with m1's scale

    self.assertEqual(m1.forward(Tensor([5])).item(), 10)

    # ACTUAL FOOTGUN: m2 uses m1's captured graph!
    # The JIT decorator is shared at the class level via __get__
    # so m2.forward uses the same captured kernel as m1.forward
    self.assertEqual(m2.forward(Tensor([5])).item(), 10)  # should be 15!


class TestRealizingSideEffects(unittest.TestCase):
  """
  FOOTGUN: Operations with side effects (like print or file writes via numpy)
  only happen during warmup/capture, not during JIT execution.

  EXPECTED BEHAVIOR: Document clearly that side effects are not replayed.
  """

  def test_side_effects_during_warmup_only(self):
    call_count = [0]

    @TinyJit
    def f(x):
      call_count[0] += 1
      return (x * 2).realize()

    # warmup
    f(Tensor([1]))
    # capture
    f(Tensor([2]))

    self.assertEqual(call_count[0], 2)  # called during warmup and capture

    # jit execution
    f(Tensor([3]))
    f(Tensor([4]))
    f(Tensor([5]))

    # ACTUAL: function body not executed during JIT replay
    self.assertEqual(call_count[0], 2)  # still 2, not 5!


class TestInputMutation(unittest.TestCase):
  """
  FOOTGUN: Mutating inputs inside JIT via assign affects the original tensor.
  This is intentional but can be surprising.

  Without JIT: Same behavior.
  With JIT: Same behavior (consistent).

  This is actually consistent, documenting for clarity.
  """

  def test_mutation_consistent(self):
    @TinyJit
    def f(x):
      x += 1
      x.realize()
      return x

    a = Tensor([0]).contiguous().realize()
    for i in range(5):
      result = f(a)

    # input is mutated in place
    self.assertEqual(a.item(), 5)


if __name__ == '__main__':
  unittest.main()
