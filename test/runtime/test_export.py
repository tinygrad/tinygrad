import multiprocessing, unittest
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
from tinygrad import Tensor, Variable, Context
from tinygrad.renderer import Compiler
from tinygrad.uop import Ops
from tinygrad.uop.uir import render_uir, parse_uir

def compile_uir(src:str) -> str:
  with Context(VALIDATE_WITH_CPU=1): return render_uir(Tensor.compile(parse_uir(src)))

class TestExportRun(unittest.TestCase):
  def test_export_compile_run(self):
    def export(xs, name, n):
      x, v = Tensor(xs).realize(), Variable(name, 1, 4).bind(n)
      out = ((x + 1).contiguous()[:v] * v).sum()
      return out, *out.export()

    a, fn, first = export([1, 2, 3, 4], "rows", 2)
    # The compiler gets only the graph; the runner gets only the compiled artifact and its arguments.
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as compiler:
      linear = parse_uir(compiler.submit(compile_uir, render_uir(fn)).result())
    self.assertFalse(any(u.op is Ops.BUFFER and u.arg.buffer is not None for u in linear.toposort(enter_calls=True)))

    b, same_fn, second = export([10, 20, 30, 40], "cols", 3)
    self.assertIs(fn, same_fn)
    with patch("tinygrad.engine.realize.lower_and_compile", side_effect=AssertionError("compiled during run")), \
         patch.object(Compiler, "compile_cached", side_effect=AssertionError("compiled during link")):
      Tensor.run(linear, second)
      Tensor.run(linear, first)
    self.assertEqual((a.item(), b.item()), (2*(2+3), 3*(11+21+31)))

  def test_link_scalar_extern(self):
    scale = Variable("scale", 1, 4)
    out = Tensor([1, 2, 3]).realize() * scale.bind(2)
    fn, externs = out.export()
    compiled = Tensor.compile(fn)
    for n in (2, 3):
      Tensor.run(compiled, tuple(scale.bind(n) if x.is_bound_var else x for x in externs))
      self.assertEqual(out.tolist(), [n, 2*n, 3*n])

  def test_run_checks_externs(self):
    a, b = Tensor([1., 2.]).realize(), Tensor([1., 2., 3.]).realize()
    fn, externs = Tensor.export((a + 1).contiguous())
    _, other = Tensor.export((b + 1).contiguous())
    linear = Tensor.compile(fn)
    with self.assertRaisesRegex(RuntimeError, "must match its EXTERN"): Tensor.run(linear, other)
    with self.assertRaisesRegex(RuntimeError, "got nothing"): Tensor.run(linear, externs[:1])

if __name__ == "__main__":
  unittest.main()
