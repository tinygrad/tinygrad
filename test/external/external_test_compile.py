import re, tempfile, unittest
from pathlib import Path
import numpy as np
from extra.benchmark_pickle import make_inputs
from tinygrad.helpers import fetch, getenv
from tinygrad.nn.compile import load_pickle
from tinygrad.nn.state import get_parameters
from tinygrad.uop.ops import Ops


class TestConfiguredCompile(unittest.TestCase):
  def test_warp_layouts(self):
    from tinygrad import Tensor
    from tinygrad.nn.compile_warp import NV12Frame, compile_warp
    frame = NV12Frame(8, 6, 12, 8, 4, 144)
    data = np.zeros((12, 12), dtype=np.uint8)
    data[:6, :8] = np.arange(48, dtype=np.uint8).reshape(6, 8)
    data[8:, 0:8:2], data[8:, 1:8:2] = 100, 200
    image = Tensor(data.reshape(-1)).realize()
    transform = np.eye(3, dtype=np.float32)
    luma = compile_warp(frame, (4, 4), layout='luma', border_fill=16, benchmark_runs=1)['variants']['default']['run']
    np.testing.assert_array_equal(luma(input_frame=image, M_inv=Tensor(transform, device='NPY')).numpy(), data[:4, :4].reshape(1, 16))
    transform[0, 2] = 1000
    np.testing.assert_array_equal(luma(input_frame=image, M_inv=Tensor(transform, device='NPY')).numpy(), np.full((1, 16), 16, dtype=np.uint8))
    yuv = compile_warp(frame, (4, 4), layout='yuv420', benchmark_runs=1)['variants']['default']['run']
    expected = [[[0, 2], [16, 18]], [[8, 10], [24, 26]], [[1, 3], [17, 19]], [[9, 11], [25, 27]], [[100]*2]*2, [[200]*2]*2]
    np.testing.assert_array_equal(yuv(input_frame=image, M_inv=Tensor(np.eye(3, dtype=np.float32), device='NPY')).numpy(), expected)

  def test_packed_history(self):
    import onnx
    from tinygrad import Tensor
    from tinygrad.nn.compile_onnx import compile_onnx
    graph = onnx.helper.make_graph([onnx.helper.make_node('Add', ['sequence', 'sequence'], ['output'])], 'history',
      [onnx.helper.make_tensor_value_info('sequence', onnx.TensorProto.FLOAT16, [1, 3, 2])],
      [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT16, [1, 3, 2])])
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'history.onnx'
      onnx.save(onnx.helper.make_model(graph), path)
      for reduce in ['sample', 'max']:
        with self.subTest(reduce=reduce):
          config = {'inputs': {'sequence': {'source': 'current', 'history': {'axis': 1, 'stride': 2, 'delay': 1, 'reduce': reduce}}},
                    'pack': ['current']}
          variant = compile_onnx(path, configs={'test': config}, float32=True, benchmark_runs=1)['variants']['test']
          arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype, _) in variant['input_specs'].items()}
          inputs = {name: Tensor(arrays[name], device=device).realize() for name, (_, _, device) in variant['input_specs'].items()}
          history = np.zeros((6, 2), dtype=np.float32)
          for value in range(1, 9):
            arrays['packed_inputs'].view(np.float32)[:] = value
            history = np.concatenate([history[1:], np.full((1, 2), value, dtype=np.float32)])
            expected = history[::2] if reduce == 'sample' else history.reshape(3, 2, 2).max(1)
            np.testing.assert_array_equal(variant['run'](**inputs).numpy(), expected[None] * 2)


@unittest.skipUnless(getenv("MODEL_PKL", ""), "requires an artifact from python -m tinygrad.nn.compile_onnx")
class TestCompiledModel(unittest.TestCase):
  def setUp(self):
    with open(getenv("MODEL_PKL", ""), 'rb') as f: self.variant = load_pickle(f, out_of_band=bool(getenv("PICKLE_OOB")))['variants']['default']
    self.model = self.variant['run']

  def test_inputs(self):
    def run(seed): return [t.numpy().copy() for t in get_parameters(self.model(**make_inputs(self.variant, seed)))]
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
      inputs = make_inputs(self.variant, seed)
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
