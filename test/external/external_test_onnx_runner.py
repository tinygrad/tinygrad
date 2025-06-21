import unittest, onnx, tempfile
from tinygrad import dtypes
from tinygrad.frontend.onnx import OnnxRunner, onnx_load
from tinygrad.device import is_dtype_supported
from extra.onnx import data_types
from hypothesis import given, settings, strategies as st

class TestOnnxRunnerDtypes(unittest.TestCase):
  def _test_input_spec_dtype(self, onnx_data_type, tinygrad_dtype):
    input_tensor = onnx.helper.make_tensor_value_info('input', onnx_data_type, ())
    output_tensor = onnx.helper.make_tensor_value_info('output', onnx_data_type, ())
    node = onnx.helper.make_node('Identity', inputs=['input'], outputs=['output'])
    graph = onnx.helper.make_graph([node], 'identity_test', [input_tensor], [output_tensor])
    model = onnx.helper.make_model(graph)
    tmp = tempfile.NamedTemporaryFile()
    tmp.close()
    onnx.save(model, tmp.name)
    model = onnx_load(tmp.name)
    runner = OnnxRunner(model)
    self.assertEqual(len(runner.graph_inputs), 1)
    self.assertEqual(runner.graph_inputs['input'].dtype, tinygrad_dtype)

  def _test_initializer_dtype(self, onnx_data_type, tinygrad_dtype):
    initializer = onnx.helper.make_tensor('initializer', onnx_data_type, (), [1])
    input_tensor = onnx.helper.make_tensor_value_info('input', onnx_data_type, ())
    output_tensor = onnx.helper.make_tensor_value_info('output', onnx_data_type, ())
    node = onnx.helper.make_node('Identity', inputs=['input'], outputs=['output'])
    graph = onnx.helper.make_graph([node], 'identity_test', [input_tensor], [output_tensor], [initializer])
    model = onnx.helper.make_model(graph)
    tmp = tempfile.NamedTemporaryFile()
    tmp.close()
    onnx.save(model, tmp.name)
    model = onnx_load(tmp.name)
    runner = OnnxRunner(model)
    self.assertEqual(len(runner.graph_inputs), 1)
    self.assertEqual(runner.graph_values['initializer'].dtype, tinygrad_dtype)

  def _test_tensor_attribute_dtype(self, onnx_data_type, tinygrad_dtype):
    output_tensor = onnx.helper.make_tensor_value_info('output', onnx_data_type, ())
    value_tensor = onnx.helper.make_tensor('value', onnx_data_type, (), [1])
    node = onnx.helper.make_node('Constant', inputs=[], outputs=['output'], value=value_tensor)
    graph = onnx.helper.make_graph([node], 'attribute_test', [], [output_tensor])
    model = onnx.helper.make_model(graph)
    tmp = tempfile.NamedTemporaryFile()
    tmp.close()
    onnx.save(model, tmp.name)
    model = onnx_load(tmp.name)
    runner = OnnxRunner(model)
    self.assertEqual(runner.graph_nodes[0].opts['value'].dtype, tinygrad_dtype)

  @settings(deadline=1000) # TODO investigate unreliable timing
  @given(onnx_data_type=st.sampled_from(list(data_types.keys())))
  def test_dtype_spec(self, onnx_data_type):
    tinygrad_dtype = data_types[onnx_data_type]
    if not is_dtype_supported(tinygrad_dtype):
      tinygrad_dtype = dtypes.default_int if dtypes.is_int(tinygrad_dtype) else dtypes.default_float

    self._test_input_spec_dtype(onnx_data_type, tinygrad_dtype)
    self._test_initializer_dtype(onnx_data_type, tinygrad_dtype)
    self._test_tensor_attribute_dtype(onnx_data_type, tinygrad_dtype)

if __name__ == '__main__':
  unittest.main()