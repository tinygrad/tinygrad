import unittest
from tinygrad import Tensor, Variable
from tinygrad.uop.uir import render_uir, parse_uir
from tinygrad.codegen import create_linear

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

  def test_run_checks_externs(self):
    a, b = Tensor([1., 2.]).realize(), Tensor([1., 2., 3.]).realize()
    fn, externs = Tensor.export((a + 1).contiguous())
    _, other = Tensor.export((b + 1).contiguous())
    linear = create_linear(fn)
    with self.assertRaisesRegex(RuntimeError, "must match its EXTERN"): Tensor.run(linear, other)
    with self.assertRaisesRegex(RuntimeError, "got nothing"): Tensor.run(linear, externs[:1])

if __name__ == "__main__":
  unittest.main()
