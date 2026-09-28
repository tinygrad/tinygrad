import unittest
from tinygrad import Tensor, UOp, Device
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import Ops, KernelInfo
from tinygrad.viz.serve import get_render
from test.null.test_viz import needs_tracked_pm, save_viz

@unittest.skipUnless(Device.DEFAULT != "NULL" and isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage), "requires a C-style compiler")
class TestViz(unittest.TestCase):
  @needs_tracked_pm
  def test_view_source(self):
    def custom_fn(X:UOp):
      X = X.flatten()
      i = UOp.range(X.numel(), 0)
      custom_op = UOp(Ops.CUSTOMI, src=(X[i],), arg=("{} + undeclared_name", X.dtype))
      return X[i].store(custom_op).end(i).sink(arg=KernelInfo(name=f"custom_fn_{X.numel()}"))
    x = Tensor.custom_kernel(Tensor.empty(1), fxn=custom_fn)[0]
    with save_viz() as viz:
      with self.assertRaises(Exception) as e:
        x.realize()
    lst = viz.list_items()
    codegen_idx = len(lst)-1
    steps = lst[codegen_idx]["steps"]
    lin_idx = next((i for i,s in enumerate(steps) if s["name"] == "View UOp List"), None)
    src_idx = next((i for i,s in enumerate(steps) if s["name"] == "View Source"), None)
    bin_idx = next((i for i,s in enumerate(steps) if s["name"] == "View Disassembly"), None)
    assert all(i is not None for i in [lin_idx, src_idx, bin_idx]), f"linear, source and disasm must be visible in {steps}"
    # Ops.LINEAR renders
    lin_render = get_render(viz.data, steps[lin_idx]["query"])["src"]
    self.assertIn("Ops.SINK", lin_render)
    self.assertIn("Ops.CUSTOMI", lin_render)
    # Ops.SOURCE renders
    src_render = get_render(viz.data, steps[src_idx]["query"])["src"]
    self.assertIn("undeclared_name", src_render)
    # Ops.BINARY shows the error message since compile failed
    bin_render = get_render(viz.data, steps[bin_idx]["query"])["src"]
    self.assertIn(type(e.exception).__name__, bin_render)

if __name__ == '__main__':
  unittest.main()
