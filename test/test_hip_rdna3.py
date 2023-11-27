#!/usr/bin/env python
import unittest
from tinygrad import Tensor
from tinygrad.helpers import dtypes, getenv
from tinygrad.ops import LoadOps
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.renderer.hip import HIPRenderer
from examples.beautiful_mnist import Model as MNIST
from examples.hlb_cifar10 import SpeedyResNet
import extra.hip_wrapper as hip


def check_rdna3_compilation(result:Tensor):
  sched = result.lazydata.schedule()

  for item in sched:
    if isinstance(item.ast.op, LoadOps): continue
    linearizer = Linearizer(item.ast, opts=LinearizerOptions(device="HIP"))
    linearizer.linearize()

    rendered = HIPRenderer("foobar", linearizer.uops)[0]

    prog = hip.hiprtcCreateProgram(rendered, "<null>", [], [])
    # https://llvm.org/docs/AMDGPUUsage.html#amd-gcn-gfx11-rdna3
    hip.hiprtcCompileProgram(prog, [f'--offload-arch=gfx1100'])
    code = hip.hiprtcGetCode(prog)
    assert len(code) > 0, "compilation to gfx1100 failed"


class TestHIPCompilationRDNA(unittest.TestCase):
  def test_compile_hip_mnist(self):
    if not getenv("HIP"): raise unittest.SkipTest("testing HIP->rdna3 compilation needs HIP=1")
    model = MNIST()

    input = Tensor.rand(512,1,28,28)
    check_rdna3_compilation(model(input))

  def test_compile_hip_speedyresnet(self):
    if not getenv("HIP"): raise unittest.SkipTest("testing HIP->rdna3 compilation needs HIP=1")
    W = Tensor.rand(12,3,2,2)
    model = SpeedyResNet(W)

    input = Tensor.rand(512,3,32,32)
    check_rdna3_compilation(model(input))

  @unittest.expectedFailure
  def test_compile_hip_speedyresnet_hf(self):
    if not getenv("HIP"): raise unittest.SkipTest("testing HIP->rdna3 compilation needs HIP=1")

    Tensor.default_type = dtypes.float16

    W = Tensor.rand(12,3,2,2)
    model = SpeedyResNet(W)

    input = Tensor.rand(512,3,32,32)
    check_rdna3_compilation(model(input))

if __name__ == "__main__":
  unittest.main()