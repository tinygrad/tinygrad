import sys
import numpy as np
from typing import Dict, Type
from tinygrad.ast import ASTKernel
from tinygrad.llops.ops_cpu import CPUBuffer
from tinygrad.ops import DeviceBuffer
from tinygrad.lazy import realize_buffers

in_test = False
test_cnt = 0
def test_ast_kernel(k:ASTKernel, device:Type[DeviceBuffer]=CPUBuffer):
  global in_test, test_cnt
  if in_test: return
  in_test = True
  print("testing AST", test_cnt)
  test_cnt += 1
  cpubufs : Dict[DeviceBuffer, DeviceBuffer] = {x:device.fromCPU(x.toCPU()) for x in k.bufs}
  real_out = cpubufs[k.bufs[0]].toCPU()
  assert hasattr(device, 'exec_ast')
  test_out = device.exec_ast(realize_buffers(cpubufs, k.ast)).toCPU()
  if not np.allclose(real_out, test_out, atol=1e-4):
    print("MISMATCH")
    print(k.print())
    sys.tracebacklimit = 0
    np.testing.assert_allclose(real_out, test_out)
  in_test = False