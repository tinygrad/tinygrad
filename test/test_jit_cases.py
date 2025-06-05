import unittest
from tinygrad import TinyJit, Tensor, Device, dtypes

# The JIT functions as a "capturing" JIT.
# Whatever kernels ran in the JIT the second run through the function will be the kernels that will run from then on.
# Explicit inputs to the function are updated in the JIT graph to the new inputs.

# JITs have four tensor types
#  1. Tensors that are explicit in the input, aka what's passed in. TODO: support lists/dicts/classes, anything get_state works on
#  2. Tensors that are explicit in the output, aka what's returned. TODO: same as above
#  3. Tensors that are implicit in the input as a closure.
#  4. Tensors that are implicit in the output because they were assigned to and realized.

# explicit inputs and outputs are realized on their way in and out of the JIT
# there's a whole bunch of edge cases and weirdness here that needs to be tested and clarified.

class TestJitCases(unittest.TestCase):
  def test_explicit(self):
    # this function has an explicit input and an explicit output
    @TinyJit
    def f(x:Tensor):
      ret:Tensor = x*2
      return ret

    for i in range(5):
      out = f(Tensor([i]))
      self.assertEqual(out.item(), i*2)

  def test_implicit_input(self):
    # x is the implicit input (like a weight)
    x = Tensor([0])

    # this function has an implicit input and an explicit output
    @TinyJit
    def f():
      ret:Tensor = x*2
      return ret

    for i in range(5):
      # NOTE: this must be realized here, otherwise the update doesn't happen
      # if we were explicitly tracking the implicit input Tensors, we might not need this realize
      x.assign(Tensor([i])).realize()
      out = f()
      self.assertEqual(out.item(), i*2)

  def test_implicit_output(self):
    # out is the implicit output (it's assigned to)
    out = Tensor([0])

    # this function has an explicit input and an implicit output
    @TinyJit
    def f(x:Tensor):
      # NOTE: this must be realized here
      # if we were explicitly tracking the implicit output Tensors, we might not need this realize
      out.assign(x*2).realize()

    for i in range(5):
      f(Tensor([i]))
      self.assertEqual(out.item(), i*2)

  def test_implicit_io(self):
    # x is the implicit input (like a weight)
    # out is the implicit output (it's assigned to)
    x = Tensor([0])
    out = Tensor([0])

    # this function has an implicit input and an implicit output
    @TinyJit
    def f():
      out.assign(x*2).realize() # NOTE: this must be realized here

    for i in range(5):
      x.assign(Tensor([i])).realize()
      f()
      self.assertEqual(out.item(), i*2)

def f16_is_unsupported():
  assert (Tensor([1], dtype=dtypes.float32) + 1).tolist() == [2], "float32 kernel failed, there is an unexpected problem"
  try: (Tensor([1], dtype=dtypes.float16) + 1).realize()
  except RuntimeError: return True
  return False

class DocumentJitRealizeCases(unittest.TestCase):
  # NOTE: These tests document behavior that's intended, but that limits what can be done by a programmer, due to JIT capture depending on realize.

  def test_capture_mutates_implicit_input(self):
    """
    If we want to export this function, and we want `w` to be [-2.0] when exported, we'll have to add copyin steps after JIT capture.
    If JIT capture didn't depend on realize, then you could export the graph with `w` having a value of [-2.0].
    You can't get `w` = [-2.0] by jitting the function starting with `w` = Tensor([(-2.0) ** (1/4)]), because complex numbers aren't supported.
    """
    w = Tensor([-2.0])

    @TinyJit
    def f():
      w.assign(w ** 2).realize()

    for _ in range(2): f()
    self.assertEqual(w.tolist(), [16.0])

  def test_cannot_capture_closure(self):
    """
    You can't JIT capture the graph defined in `x` when seen from `f` as a closure.
    The first call to TinyJit realizes `x` globally, so on the second call to TinyJit, `x` is a Ops.BUFFER UOp, not the compute graph we specified.
    """
    x = Tensor([2.0]) ** 2
    x.kernelize()

    @TinyJit
    def f():
      return x

    with self.assertRaisesRegex(AssertionError, "didn't JIT anything!"):
      for _ in range(2): f()

  @unittest.skipUnless(Device.DEFAULT=="WEBGPU" and f16_is_unsupported(), "If realize is necessary, you can't capture kernels that won't run locally")
  def test_capture_unsupported_dtype_kernel(self):
    """
    Currently you can't export f16 WebGPU models on dawn/Vulkan/NVIDIA, because f16 isn't supported in that stack.
    See: https://issues.chromium.org/issues/42251215
    Model export depends on JIT capture, and JIT capture depends on realize, which runs kernels locally.
    But even if I can't run f16 kernels locally on dawn/Vulkan/NVIDIA, if I could export the kernels without running them, then the model would still
    work on WebGPU on windows (D3D backend) and mac/iphone (Metal backend).
    """
    @TinyJit
    def f():
      return (Tensor([1], dtype=dtypes.float16) + 1)

    with self.assertRaisesRegex(RuntimeError, "f16"):
      for _ in range(2): f()

if __name__ == '__main__':
  unittest.main()
