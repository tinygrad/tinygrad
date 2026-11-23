import multiprocessing, unittest
from concurrent.futures import ProcessPoolExecutor
from tinygrad import Context, Tensor, Variable
from tinygrad.uop.uir import render_uir, parse_uir
from tinygrad.codegen import create_linear
from tinygrad.engine.realize import compile_linear, link_linear, run_linear

@Context(ALLOW_DEVICE_USAGE=0)
def compile_uir(src:str) -> str: return render_uir(Tensor.compile(parse_uir(src)))

class TestExportRun(unittest.TestCase):
  def test_export_compile_run(self):
    # export -> external compile -> run, the function and the LINEAR cross the wire as UIR
    def model(xs, n):
      v = Variable("v", 1, 4).bind(n)
      return (Tensor(xs).realize() + 1).contiguous()[:v].contiguous() * v
    a, b = model([1., 2., 3., 4.], 3), model([10., 20., 30., 40.], 2)
    fn, first = a.export()
    _, second = b.export()
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as compiler:
      linear = parse_uir(compiler.submit(compile_uir, render_uir(fn)).result())
    Tensor.run(linear, first)
    Tensor.run(linear, second)
    self.assertEqual(a.shrink(((0, 3),)).tolist(), [6., 9., 12.])
    self.assertEqual(b.shrink(((0, 2),)).tolist(), [22., 42.])

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
    linear = Tensor.compile(fn)
    with self.assertRaisesRegex(RuntimeError, "must match its EXTERN"): Tensor.run(linear, other)
    with self.assertRaisesRegex(RuntimeError, "got nothing"): Tensor.run(linear, externs[:1])

if __name__ == "__main__":
  unittest.main()
