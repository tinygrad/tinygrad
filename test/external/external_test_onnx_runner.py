import unittest, onnx, tempfile
# from tinygrad import dtypes
from tinygrad.frontend.onnx import OnnxRunner, onnx_load
# from tinygrad.device import is_dtype_supported
from extra.onnx import data_types
from hypothesis import given, strategies as st
# import numpy as np

class TestOnnxRunner(unittest.TestCase):
  @given(onnx_data_type=st.sampled_from(list(data_types.keys())))
  def test_dtype_spec(self, onnx_data_type):
    tinygrad_dtype = data_types[onnx_data_type]

    # test input spec dtype
    input_tensor = onnx.helper.make_tensor_value_info('input', onnx_data_type, ())
    output_tensor = onnx.helper.make_tensor_value_info('output', onnx_data_type, ())
    node = onnx.helper.make_node('Identity', inputs=['input'], outputs=['output'])
    graph = onnx.helper.make_graph([node], 'identity_test', [input_tensor], [output_tensor])
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)
    with tempfile.NamedTemporaryFile() as tmp:
      onnx.save(model, tmp.name)
      tmp.flush()
      model = onnx_load(tmp.name)
      runner = OnnxRunner(model)
    self.assertEqual(len(runner.graph_inputs), 1)
    self.assertEqual(runner.graph_inputs['input'].dtype, tinygrad_dtype)

    # test initializer dtype
    initializer = onnx.helper.make_tensor('initializer', onnx_data_type, (), [1.0])
    graph = onnx.helper.make_graph([node], 'identity_test', [input_tensor], [output_tensor], [initializer])
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)
    with tempfile.NamedTemporaryFile() as tmp:
      onnx.save(model, tmp.name)
      tmp.flush()
      model = onnx_load(tmp.name)
      runner = OnnxRunner(model)
    self.assertEqual(len(runner.graph_inputs), 1)
    self.assertEqual(runner.graph_inputs['input'].dtype, tinygrad_dtype)
    self.assertEqual(runner.graph_values['initializer'].dtype, tinygrad_dtype)

    # test attribute dtype
    value_tensor = onnx.helper.make_tensor('value', onnx_data_type, (), [1.0])
    node = onnx.helper.make_node('Constant', inputs=[], outputs=['output'], value=value_tensor)
    graph = onnx.helper.make_graph([node], 'attribute_test', [], [output_tensor])
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)
    with tempfile.NamedTemporaryFile() as tmp:
      onnx.save(model, tmp.name)
      tmp.flush()
      model = onnx_load(tmp.name)
      runner = OnnxRunner(model)
    self.assertIn('output', runner.graph_values)
    self.assertEqual(runner.graph_values['output'].dtype, tinygrad_dtype)

if __name__ == '__main__':
  unittest.main()