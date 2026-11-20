import unittest
from tinygrad import Tensor, Variable
from tinygrad.uop.uir import render_uir, parse_uir
from tinygrad.codegen import create_linear
from tinygrad.engine.realize import compile_linear, link_linear, run_linear

class TestExportRun(unittest.TestCase):
  def test_export_compile_run(self):
    # export -> external compile -> run, the function and the LINEAR cross the wire as UIR
    a, b = Tensor([1., 2., 3., 4.]).realize(), Tensor([10., 20., 30., 40.]).realize()
    v = Variable("v", 1, 4).bind(3)
    out = (a * b + 1).contiguous()[:v].contiguous() * v
    fn, externs = Tensor.export(out)
    linear = parse_uir(render_uir(create_linear(parse_uir(render_uir(fn)))))
    Tensor.run(linear, externs)
    self.assertEqual(out.shrink(((0, 3),)).tolist(), [33., 123., 273.])

  def test_link_scalar_extern(self):
    scale = Variable("scale", 1, 4)
    out = Tensor([1, 2, 3]).realize() * scale.bind(2)
    fn, externs = out.export()
    compiled = compile_linear(create_linear(fn))
    for n in (2, 3):
      inputs = [scale.bind(n) if x.is_bound_var else x for x in externs]
      run_linear(link_linear(compiled, inputs), input_uops=inputs, jit=True)
      self.assertEqual(out.tolist(), [n, 2*n, 3*n])

  def test_run_checks_externs(self):
    a, b = Tensor([1., 2.]).realize(), Tensor([1., 2., 3.]).realize()
    fn, externs = Tensor.export((a + 1).contiguous())
    _, other = Tensor.export((b + 1).contiguous())
    linear = create_linear(fn)
    with self.assertRaisesRegex(RuntimeError, "must match its EXTERN"): Tensor.run(linear, other)
    with self.assertRaisesRegex(RuntimeError, "got nothing"): Tensor.run(linear, externs[:1])

if __name__ == "__main__":
  unittest.main()
