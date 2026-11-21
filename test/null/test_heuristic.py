import unittest
from tinygrad import Tensor
from tinygrad.codegen.opt import OptOps
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.renderer import Target
from tinygrad.renderer.llvmir import AMDLLVMRenderer

def group(outs:int, n:int) -> int:
  ast = Tensor.empty(outs, n, device="NULL").sum(1).schedule_linear().src[-1].src[0]
  k = hand_coded_optimizations(Scheduler(ast, AMDLLVMRenderer(Target("AMD", arch="gfx1201"))))
  return next(o.arg[0] for o in k.applied_opts if o.op is OptOps.SPLIT and len(o.arg) == 3 and o.arg[2])

class TestGroupedReduce(unittest.TestCase):
  def test_few_outputs_group_more(self):  # a long reduce of few outputs isn't 16 threads looping over it
    self.assertEqual(group(5, 7168), 256)
    self.assertEqual(group(64, 7168), 64)
  def test_many_outputs_group_16(self):  # (enough threads already: a wider group is slower)
    self.assertEqual(group(448, 7168), 16)
    self.assertEqual(group(2048, 7168), 16)
  def test_short_reduce_group_16(self): self.assertEqual(group(32, 128), 16)  # (a per head norm: 16 threads of 8 each, not 128 of 1)

if __name__ == "__main__":
  unittest.main()
