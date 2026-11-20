import multiprocessing, unittest
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
from tinygrad import Tensor, Variable, Context, Device
from tinygrad.renderer import Compiler
from tinygrad.uop import Ops
from tinygrad.uop.uir import render_uir, parse_uir

def export(xs, name, n):
  # Bind host storage so preparing inputs doesn't warm up the runner's accelerator queues.
  x, v = Tensor(xs, device="CPU").realize().to(Device.DEFAULT), Variable(name, 1, 4).bind(n)
  out = ((x + Variable("delta", 1, 4).bind(1)).contiguous()[:v] * v).sum().to("CPU")
  return out, *out.export()

def run_uir(src):
  linear = parse_uir(src)
  a, fn, first = export([1, 2, 3, 4], "rows", 2)
  b, same_fn, second = export([10, 20, 30, 40], "cols", 3)
  assert fn is same_fn
  with patch("tinygrad.engine.realize.lower_and_compile", side_effect=AssertionError("compiled during run")), \
       patch.object(Compiler, "compile_cached", side_effect=AssertionError("compiled during link")):
    Tensor.run(linear, second)
    Tensor.run(linear, first)
    vals = a.item(), b.item()
    # Even two scalars with the same runtime name bind independently by slot.
    Tensor.run(linear, tuple(Variable("n", 1, 4).bind(3 if x.expr == "rows" else 1) if x.is_bound_var else x for x in first))
  return *vals, a.item()

class TestExportRun(unittest.TestCase):
  def test_export_compile_run(self):
    _, fn, _ = export([0]*4, "rows", 2)
    with Context(VALIDATE_WITH_CPU=1): linear = Tensor.compile(parse_uir(render_uir(fn)))
    self.assertFalse(any(u.op is Ops.BUFFER and u.arg.buffer is not None for u in linear.toposort(enter_calls=True)))
    src = render_uir(linear)
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as runner:
      result = runner.submit(run_uir, src)
      if Device.DEFAULT.split(":")[0] == "NV":
        with self.assertRaisesRegex(RuntimeError, "NV batches must be compiled and run in the same process"): result.result()
        self.assertEqual(run_uir(src), (10, 189, 27))
      else: self.assertEqual(result.result(), (10, 189, 27))

  def test_run_checks_externs(self):
    a, b = Tensor([1., 2.]).realize(), Tensor([1., 2., 3.]).realize()
    fn, externs = Tensor.export((a + 1).contiguous())
    _, other = Tensor.export((b + 1).contiguous())
    linear = Tensor.compile(fn)
    with self.assertRaisesRegex(RuntimeError, "must match its EXTERN"): Tensor.run(linear, other)
    with self.assertRaisesRegex(RuntimeError, "got nothing"): Tensor.run(linear, externs[:1])

if __name__ == "__main__":
  unittest.main()
