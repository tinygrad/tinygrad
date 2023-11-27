#!/usr/bin/env python
import unittest
from tinygrad import Tensor, nn
from tinygrad.helpers import getenv
from tinygrad.ops import LoadOps
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.renderer.hip import HIPRenderer
from examples.beautiful_mnist import Model
import extra.hip_wrapper as hip


class TestHIPCompilationRDNA(unittest.TestCase):
  def test_compile_hip(self):
    if not getenv("HIP"): raise unittest.SkipTest("testing HIP->rdna3 compilation needs HIP=1")

    model = Model()

    input = Tensor.rand(512,1,28,28, device="CPU").numpy()
    result = model(Tensor(input))
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


if __name__ == "__main__":
  unittest.main()