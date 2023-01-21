import sys
import numpy as np
from typing import Dict
from tinygrad.ast import ASTKernel
from tinygrad.llops.ops_cpu import CPUBuffer
from tinygrad.ops import DeviceBuffer
from tinygrad.lazy import realize_buffers

in_test = False
def test_ast(k:ASTKernel):
  global in_test
  if in_test: return
  in_test = True
  print("testing AST")
  cpubufs : Dict[DeviceBuffer, CPUBuffer] = {x:CPUBuffer.fromCPU(x.toCPU()) for x in k.bufs}
  real_out = cpubufs[k.bufs[0]]
  test_out = CPUBuffer.exec_ast(realize_buffers(cpubufs, k.ast))
  if not np.allclose(real_out, test_out, atol=1e-4):
    print("MISMATCH")
    print(k.print())
    sys.tracebacklimit = 0
    np.testing.assert_allclose(real_out, test_out)
  in_test = False