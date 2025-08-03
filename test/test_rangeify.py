import unittest
from tinygrad import Tensor, Device
from tinygrad.uop.ops import KernelInfo
from tinygrad.opt.kernel import Opt, OptOps
from tinygrad.engine.realize import get_program

def with_opts(c:Tensor, opts_to_apply:list[Opt]):
  s = c.schedule()[-1]
  program = get_program(s.ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply))), Device.default.renderer)
  print(program.src)

class TestRangeify(unittest.TestCase):
  def test_upcast(self):
    a = Tensor.empty(4, 4)
    b = Tensor.empty(4, 4)
    c = a + b
    with_opts(c, [Opt(op=OptOps.UPCAST, axis=1, arg=4)])

  def test_upcast_sum(self):
    a = Tensor.empty(4, 4)
    b = a.sum(axis=1)
    with_opts(b, [Opt(op=OptOps.UPCAST, axis=0, arg=4)])

if __name__ == '__main__':
  unittest.main()