import re, unittest
import numpy as np
from extra.benchmark_pickle import make_inputs
from tinygrad.helpers import fetch, getenv
from tinygrad.nn.compile import load_pickle
from tinygrad.nn.state import get_parameters
from tinygrad.uop.ops import Ops


@unittest.skipUnless(getenv("MODEL_PKL", ""), "requires an artifact from python -m tinygrad.nn.compile")
class TestCompiledModel(unittest.TestCase):
  def setUp(self):
    with open(getenv("MODEL_PKL", ""), 'rb') as f: self.model = load_pickle(f, out_of_band=bool(getenv("PICKLE_OOB")))

  def test_inputs(self):
    def run(seed): return [t.numpy().copy() for t in get_parameters(self.model(**make_inputs(self.model, seed)))]
    original, changed, repeated = run(100), run(101), run(100)
    for before, after, again in zip(original, changed, repeated, strict=True):
      self.assertTrue(np.isfinite(before).all() and np.isfinite(after).all())
      np.testing.assert_array_equal(before, again)
    self.assertFalse(all(np.array_equal(a, b) for a, b in zip(original, changed, strict=True)))

  @unittest.skipUnless(getenv("MODEL_ONNX", ""), "requires the source ONNX model")
  def test_onnx(self):
    import onnxruntime as ort
    session = ort.InferenceSession(str(fetch(getenv("MODEL_ONNX", ""))), providers=['CPUExecutionProvider'])
    input_types = {v.name: v.type.removeprefix('tensor(').removesuffix(')') for v in session.get_inputs()}
    for seed in (100, 101):
      inputs = make_inputs(self.model, seed)
      reference = session.run(None, {k: v.numpy().astype({'float': 'float32', 'double': 'float64'}.get(input_types[k], input_types[k]))
                                    for k, v in inputs.items()})
      for expected, actual in zip(reference, get_parameters(self.model(**inputs)), strict=True):
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-4, rtol=1e-4)

  @unittest.skipUnless(getenv("ALLOWED_KERNEL_COUNT", -1) != -1, "requires kernel regression targets")
  def test_kernel_counts(self):
    calls = [u for u in self.model.captured.linear.toposort(gate=lambda x: x.op is not Ops.PROGRAM)
             if u.op is Ops.CALL and u.src[0].op is Ops.PROGRAM]
    read_image = gated_read_image = 0
    for call in calls:
      _, _, source, _ = call.src[0].src
      src = source.arg
      read_image += src.count("read_image")
      gated_read_image += src.count("?read_image")
      for value in re.findall(r'(val\d+)\s*=\s*read_imagef\(', src):
        if re.search(fr'[\?\:]{value}\.[xyzw]', src): gated_read_image += 1
    self.assertEqual(len(calls), getenv("ALLOWED_KERNEL_COUNT"))
    self.assertEqual(read_image, getenv("ALLOWED_READ_IMAGE"))
    self.assertEqual(gated_read_image, getenv("ALLOWED_GATED_READ_IMAGE"))


if __name__ == '__main__': unittest.main()
