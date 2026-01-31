import unittest
from tinygrad.uop.ops import UOp, Ops, dtypes, graph_rewrite
from tinygrad.renderer.isa import IselContext
from tinygrad.renderer.x86 import X86Renderer

class TestX86Schedule(unittest.TestCase):
  def schedule(self, x:UOp) -> list[UOp]:
    x = graph_rewrite(x, X86Renderer().pre_isel_matcher)
    x = graph_rewrite(x, X86Renderer().isel_matcher, IselContext(x), bottom_up=True)

  def test_hide_latency(self):
    buf = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), arg=0)
    load1 = buf.index(UOp.const(dtypes.int32, 1), ptr=True).load()
    load2 = buf.index(UOp.const(dtypes.int32, 2), ptr=True).load()
    const = UOp.const(dtypes.float32, 1)
    # short path, cheap alu
    add = load1 + const
    # long path, expensive alu
    #fmadd = UOp.alu(Ops.MULACC, load2, const, const)
    # unify the paths
    #n = self.schedule(add + fmadd)
    # load2 should be picked first as it has a longer path

  # in-order core can't issue ops with dependencies between them in a single cycle
  def test_issue_io(self): pass

  # out-of-order core can issue ops with dependencies between them in a single cycle
  def test_issue_ooo(self): pass

  # if micro ops > issue width can issue this cycle if no other micro ops were issued
  def test_issue_width_empty_cycle(self): pass

  # if micro ops were issued this cycle and issue width can't fit micro ops then they can't be issued this cycle
  def test_issue_width_non_empty_cycle(self): pass

  # test cycles advance and no op is issued until stall clears
  def test_stall(self): pass

  # test reg pressure
  def test_reg_pressure(self): pass

  # test you can issue x whose unit was reserved for y but x's unit end cycle <= y's unit start cycle
  def test_resource_cycles_no_intersection(self): pass

  # now test x's unit end cycle > y's unit start cycle, can still issue x if ooo
  def test_resource_cycles_intersection(self): pass

