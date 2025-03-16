from tinygrad.tensor import Tensor
from tinygrad.codegen.kernel import Kernel
from tinygrad.runtime.ops_gpu import CLDevice, IntelOfflineCompiler
import numpy as np
import pickle
import unittest

# TODO: skip if not intel hardware
class TestIntelOcloc(unittest.TestCase):
  def test_compare_binary(self):
    a = Tensor.ones((1, 3, 4096))
    b = Tensor.ones((1, 3, 4096))
    c = Tensor.conv2d(a,b,padding=1)

    # TODO: determine target ip version (via opencl possibel ?)

    # runtime binary 
    sched = c.schedule()
    kernel = Kernel(sched[-1].ast)
    kernel.linearize()

    # render 

    # compile 

    print("kernel uops ", kernel.uops)

    # ocloc binary 
   

if __name__ == '__main__':
  unittest.main()