#!/usr/bin/env python
import unittest
import operator
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import DEBUG, to_function_name
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.renderer.cstyle import HIPRenderer
from examples.beautiful_mnist import Model as MNIST
from examples.hlb_cifar10 import SpeedyResNet

from hypothesis import given, strategies as st, settings
settings.register_profile("my_profile", deadline=None)
settings.load_profile("my_profile")
print(settings.default)

@unittest.skipIf(Device.DEFAULT != "HIP", reason="testing HIP->rdna3 compilation needs HIP=1")
class TestHIPCompilationRDNA(unittest.TestCase):
  def test_compile_hip_mnist(self):
    model = MNIST()

    input = Tensor.rand(512,1,28,28)
    output = model(input)
    output.numpy()

  def test_compile_hip_speedyresnet(self):
    W = Tensor.rand(12,3,2,2)
    model = SpeedyResNet(W)

    input = Tensor.rand(512, 3, 32, 32)
    output = model(input)
    output.numpy()

  def test_compile_hip_speedyresnet_hf(self):
    old_default_float = dtypes.default_float
    dtypes.default_float = dtypes.float16

    W = Tensor.rand(12,3,2,2)
    model = SpeedyResNet(W)

    input = Tensor.rand(512, 3, 32, 32)
    output = model(input)
    output.numpy()

    dtypes.default_float = old_default_float

def compile_ast_to_hip(out: Tensor):
  from tinygrad.runtime.ops_hip import compile_hip

  lin = Linearizer(out.lazydata.schedule()[-1].ast)
  lin.hand_coded_optimizations()
  lin.linearize()
  code = HIPRenderer(to_function_name(lin.name), lin.uops)[0]
  if DEBUG >= 4: print(code)
  compile_hip(code)

binary_operations = [operator.add, operator.sub, operator.mul]
unary_operations = [Tensor.exp, Tensor.log, operator.neg, Tensor.sin, Tensor.sqrt, Tensor.reciprocal]
float_dtypes = [dtypes.float16, dtypes.float32]

@unittest.skipIf(Device.DEFAULT != "HIP", reason="testing HIP->rdna3 compilation needs HIP=1")
class TestHIPALUCompilation(unittest.TestCase):
  @given(st.sampled_from(unary_operations), st.sampled_from(float_dtypes))
  def test_unary_ops(self, op, dtype):
    a = Tensor.randn(4,4, dtype=dtype)
    out = op(a)
    compile_ast_to_hip(out)

  @given(st.sampled_from(binary_operations), st.sampled_from(float_dtypes))
  def test_binary_ops(self, op, dtype):
    a = Tensor.randn(4,4, dtype=dtype)
    b = Tensor.randn(4,4, dtype=dtype)
    out = op(a,b)
    compile_ast_to_hip(out)

if __name__ == "__main__":
  unittest.main()
