import unittest, onnx, tempfile
import numpy as np
from tinygrad import dtypes, Tensor
from tinygrad.uop import Ops
from tinygrad.frontend.onnx import OnnxRunner, onnx_load
from tinygrad.device import is_dtype_supported
from extra.onnx import data_types
from hypothesis import given, settings, strategies as st

# copied from test_const_folding.py
def _check_ast_count(desired_count:int, t:Tensor):
  # NOTE: this has side effect because everything can be scheduled only once
  schedule = t.schedule()
  asts = [s for s in schedule if s.ast.op is Ops.SINK]
  assert len(asts) == desired_count, f"{len(asts)} != {desired_count}"

class TestOnnxRunner(unittest.TestCase):
  def test_const_fold(self):
    inp_val = 1.0
    inp_tensor = onnx.helper.make_tensor('inp', onnx.TensorProto.FLOAT, (), [inp_val])
    shape_val = np.array([5], dtype=np.int64)
    shape_tensor = onnx.helper.make_tensor('new_shape', onnx.TensorProto.INT64, shape_val.shape, shape_val)
    expand_node = onnx.helper.make_node('Expand', inputs=['inp', 'new_shape'], outputs=['expanded_tensor'])
    exp_node = onnx.helper.make_node('Exp', inputs=['expanded_tensor'], outputs=['output'])
    final_shape = (5,)
    output_info = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, final_shape)
    graph = onnx.helper.make_graph([expand_node, exp_node], 'const_fold_expand_exp_test', [], [output_info], [inp_tensor, shape_tensor])
    model = onnx.helper.make_model(graph)
    tmp = tempfile.NamedTemporaryFile(suffix='.onnx')
    onnx.save(model, tmp.name)
    tmp.flush()
    model = onnx_load(tmp.name)
    runner = OnnxRunner(model)
    out = runner({})['output']
    _check_ast_count(0, out)

  def test_const_fold_binary_ops(self):
    inp_val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    inp = onnx.helper.make_tensor('inp', onnx.TensorProto.FLOAT, inp_val.shape, inp_val)
    const = onnx.helper.make_tensor('const', onnx.TensorProto.FLOAT, (), [0])
    node = onnx.helper.make_node('Add', inputs=['inp', 'const'], outputs=['output'])
    output_info = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, inp_val.shape)
    graph = onnx.helper.make_graph([node], 'const_fold_test', [], [output_info], [inp, const])
    model = onnx.helper.make_model(graph)
    tmp = tempfile.NamedTemporaryFile(suffix='.onnx')
    onnx.save(model, tmp.name)
    tmp.flush()
    model = onnx_load(tmp.name)
    runner = OnnxRunner(model)
    out = runner({})['output']
    _check_ast_count(0, out)

device_supported_dtypes = [odt for odt, dtype in data_types.items() if is_dtype_supported(dtype)]
device_unsupported_dtypes = [odt for odt, dtype in data_types.items() if not is_dtype_supported(dtype)]

class TestOnnxRunnerDtypes(unittest.TestCase):
  def _test_input_spec_dtype(self, onnx_data_type, tinygrad_dtype):
    input_tensor = onnx.helper.make_tensor_value_info('input', onnx_data_type, ())
    output_tensor = onnx.helper.make_tensor_value_info('output', onnx_data_type, ())
    node = onnx.helper.make_node('Identity', inputs=['input'], outputs=['output'])
    graph = onnx.helper.make_graph([node], 'identity_test', [input_tensor], [output_tensor])
    model = onnx.helper.make_model(graph)
    tmp = tempfile.NamedTemporaryFile(suffix='.onnx')
    onnx.save(model, tmp.name)
    tmp.flush()
    model = onnx_load(tmp.name)
    runner = OnnxRunner(model)
    self.assertEqual(len(runner.graph_inputs), 1)
    self.assertEqual(runner.graph_inputs['input'].dtype, tinygrad_dtype)

  def _test_initializer_dtype(self, onnx_data_type, tinygrad_dtype):
    initializer = onnx.helper.make_tensor('initializer', onnx_data_type, (2,), [1, 2])
    input_tensor = onnx.helper.make_tensor_value_info('input', onnx_data_type, ())
    output_tensor = onnx.helper.make_tensor_value_info('output', onnx_data_type, ())
    node = onnx.helper.make_node('Identity', inputs=['input'], outputs=['output'])
    graph = onnx.helper.make_graph([node], 'identity_test', [input_tensor], [output_tensor], [initializer])
    model = onnx.helper.make_model(graph)
    tmp = tempfile.NamedTemporaryFile(suffix='.onnx')
    onnx.save(model, tmp.name)
    tmp.flush()
    model = onnx_load(tmp.name)
    runner = OnnxRunner(model)
    self.assertEqual(len(runner.graph_inputs), 1)
    self.assertEqual(runner.graph_values['initializer'].dtype, tinygrad_dtype)

  def _test_node_attribute_dtype(self, onnx_data_type, tinygrad_dtype):
    output_tensor = onnx.helper.make_tensor_value_info('output', onnx_data_type, (2,))
    value_tensor = onnx.helper.make_tensor('value', onnx_data_type, (2,), [1, 2])
    node = onnx.helper.make_node('Constant', inputs=[], outputs=['output'], value=value_tensor)
    graph = onnx.helper.make_graph([node], 'attribute_test', [], [output_tensor])
    model = onnx.helper.make_model(graph)
    tmp = tempfile.NamedTemporaryFile(suffix='.onnx')
    tmp.flush()
    onnx.save(model, tmp.name)
    model = onnx_load(tmp.name)
    runner = OnnxRunner(model)
    self.assertEqual(runner.graph_nodes[0].opts['value'].dtype, tinygrad_dtype)

  @settings(deadline=3000) # TODO investigate unreliable timing
  @given(onnx_data_type=st.sampled_from(device_supported_dtypes))
  def test_supported_dtype_spec(self, onnx_data_type):
    tinygrad_dtype = data_types[onnx_data_type]
    self._test_input_spec_dtype(onnx_data_type, tinygrad_dtype)
    self._test_initializer_dtype(onnx_data_type, tinygrad_dtype)
    self._test_node_attribute_dtype(onnx_data_type, tinygrad_dtype)

  @unittest.skipUnless(device_unsupported_dtypes, "No unsupported dtypes for this device to test.")
  @settings(deadline=3000) # TODO investigate unreliable timing
  @given(onnx_data_type=st.sampled_from(device_unsupported_dtypes))
  def test_unsupported_dtype_spec(self, onnx_data_type):
    true_dtype = data_types[onnx_data_type]
    default_dtype = dtypes.default_int if dtypes.is_int(true_dtype) else dtypes.default_float
    self._test_input_spec_dtype(onnx_data_type, true_dtype)
    self._test_initializer_dtype(onnx_data_type, default_dtype)
    self._test_node_attribute_dtype(onnx_data_type, default_dtype)

if __name__ == '__main__':
  unittest.main()