import unittest
from tinygrad import TinyJit, Tensor, Device, dtypes

def f16_is_unsupported():
  assert (Tensor([1], dtype=dtypes.float32) + 1).tolist() == [2], "float32 kernel failed, there is an unexpected problem"
  try: (Tensor([1], dtype=dtypes.float16) + 1).realize()
  except RuntimeError: return True
  return False

class TestJitFailures(unittest.TestCase):

  def test_capture_mutates_implicit_input(self):
    """
    Current JIT capture requires realizing the function's computation twice, which can "ruin" our state.
    You can't just JIT this function from "two steps back" because tinygrad doesn't have complex numbers.
    When exporting this function, if we want `w` to be [-2.0] like when we defined the function, we'll have to do extra manual work to fix `w`.
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
    Currently you can't export f16 WebGPU models on dawn+Vulkan+NVIDIA, because f16 isn't supported in that stack.
    See: https://issues.chromium.org/issues/42251215
    But aside from BEAM, it shouldn't matter if the kernels won't run locally, the exported model could still work on other devices.
    """
    @TinyJit
    def f():
      return (Tensor([1], dtype=dtypes.float16) + 1)

    with self.assertRaisesRegex(RuntimeError, "f16"):
      for _ in range(2): f()

if __name__ == '__main__':
  unittest.main()
