import unittest, onnx, tempfile, pathlib
from tinygrad import dtypes, Tensor
from tinygrad.device import is_dtype_supported
from extra.onnx import data_types
from tinygrad.frontend.onnx import OnnxRunner
from hypothesis import given, strategies as st

def build_onnx(nodes, disk:bool=True, **kwargs):
  """Helper to build and return an OnnxRunner from ONNX nodes."""
  graph = onnx.helper.make_graph(nodes, 'test', kwargs.get('inputs', []), kwargs.get('outputs', []), kwargs.get('initializers', []))
  model = onnx.helper.make_model(graph)
  if disk:
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_path = pathlib.Path(tmpdir)
      model_path = tmp_path / "model.onnx"
      onnx.save(model, model_path)
      runner = OnnxRunner(model_path)
  else:
    # use the in-memory method
    runner = OnnxRunner(Tensor(model.SerializeToString(), device="PYTHON"))
  return runner

all_dtypes = list(data_types.keys())
device_supported_dtypes = {odt for odt, dtype in data_types.items() if is_dtype_supported(dtype)}

class TestOnnxRunnerDtypes(unittest.TestCase):
  """
  Initializer Tensors and attribute Tensors are Tensors internal to the ONNX model,
  so they should fallback to default dtype if device does not support.
  Input Tensors are external to the ONNX model, so they should preserve their true dtype.
  """
  def _get_expected_dtype(self, onnx_dtype: int, is_input: bool):
    true_dtype = data_types[onnx_dtype]
    # inputs always preserve their true dtype.
    if is_input:
      return true_dtype
    # supported types are always themselves.
    if onnx_dtype in device_supported_dtypes:
      return true_dtype
    # otherwise it's an unsupported dtype that's internal to the ONNX model, which should fallback to default.
    return dtypes.default_int if dtypes.is_int(true_dtype) else dtypes.default_float

  @given(onnx_dtype=st.sampled_from(all_dtypes))
  def test_input_dtype(self, onnx_dtype: int):
    expected_dtype = self._get_expected_dtype(onnx_dtype, True)
    runner = build_onnx(
        nodes=[onnx.helper.make_node('Identity', ['input'], ['output'])],
        inputs=[onnx.helper.make_tensor_value_info('input', onnx_dtype, ())],
        outputs=[onnx.helper.make_tensor_value_info('output', onnx_dtype, ())],
        disk=False)
    self.assertEqual(runner.graph_inputs['input'].dtype, expected_dtype)

  @given(onnx_dtype=st.sampled_from(all_dtypes))
  def test_initializer_dtype(self, onnx_dtype: int):
    expected_dtype = self._get_expected_dtype(onnx_dtype, False)
    runner = build_onnx(
        nodes=[onnx.helper.make_node('Identity', ['initializer'], ['output'])],
        outputs=[onnx.helper.make_tensor_value_info('output', onnx_dtype, (2,))],
        initializers=[onnx.helper.make_tensor('initializer', onnx_dtype, (2,), [1, 2])],
        disk=False)
    self.assertEqual(runner.graph_values['initializer'].dtype, expected_dtype)

  @given(onnx_dtype=st.sampled_from(all_dtypes))
  def test_node_attribute_dtype(self, onnx_dtype: int):
    expected_dtype = self._get_expected_dtype(onnx_dtype, False)
    value_tensor = onnx.helper.make_tensor('value', onnx_dtype, (2,), [1, 2])
    runner = build_onnx(
        nodes=[onnx.helper.make_node('Constant', [], ['output'], value=value_tensor)],
        outputs=[onnx.helper.make_tensor_value_info('output', onnx_dtype, (2,))],
        disk=False)
    self.assertEqual(runner.graph_nodes[0].opts['value'].dtype, expected_dtype)

if __name__ == '__main__':
  unittest.main()