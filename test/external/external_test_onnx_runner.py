import unittest, onnx, tempfile, pathlib
import numpy as np
from tinygrad import dtypes, Tensor
from tinygrad.dtype import DType
from tinygrad.uop import Ops
from tinygrad.frontend.onnx import OnnxRunner
from tinygrad.device import is_dtype_supported
from extra.onnx import data_types
from hypothesis import given, strategies as st

def check_ast_count(expected: int, tensor: Tensor):
  """Check AST node count after scheduling."""
  schedule = tensor.schedule()
  asts = [s for s in schedule if s.ast.op is Ops.SINK]
  assert len(asts) == expected, f"Expected {expected} AST nodes, got {len(asts)}"

def run_onnx(nodes, inputs=None, outputs=None, initializers=None, input_data=None):
  """Create and run ONNX model in one call."""
  graph = onnx.helper.make_graph(nodes, 'test', inputs or [], outputs or [], initializers or [])
  model = onnx.helper.make_model(graph)
  with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
    onnx.save(model, tmp.name)
    tmp.flush()
    runner = OnnxRunner(tmp.name)
    return runner, runner(input_data or {})


class TestOnnxRunner(unittest.TestCase):
  def test_const_fold_expand_exp(self):
    inp = onnx.helper.make_tensor('inp', onnx.TensorProto.FLOAT, (), [1.0])
    shape = onnx.helper.make_tensor('shape', onnx.TensorProto.INT64, (1,), [5])
    nodes = [
      onnx.helper.make_node('Expand', ['inp', 'shape'], ['expanded']),
      onnx.helper.make_node('Exp', ['expanded'], ['output'])
    ]
    outputs = [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, (5,))]
    _, results = run_onnx(nodes, outputs=outputs, initializers=[inp, shape])
    check_ast_count(0, results['output'])

  def test_const_fold_binary_add(self):
    inp = onnx.helper.make_tensor('inp', onnx.TensorProto.FLOAT, (4,), [1, 2, 3, 4])
    const = onnx.helper.make_tensor('const', onnx.TensorProto.FLOAT, (), [0])
    nodes = [onnx.helper.make_node('Add', ['inp', 'const'], ['output'])]
    outputs = [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, (4,))]
    _, results = run_onnx(nodes, outputs=outputs, initializers=[inp, const])
    check_ast_count(0, results['output'])

  def test_external_data_loading(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      tmpdir_path = pathlib.Path(tmpdir)

      external_data_file = tmpdir_path / "weights.bin"
      weights_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
      external_data_file.write_bytes(weights_data.tobytes())

      weights_tensor = onnx.TensorProto()
      weights_tensor.name = 'weights'
      weights_tensor.data_type = onnx.TensorProto.FLOAT
      weights_tensor.dims[:] = [4]

      weights_tensor.data_location = onnx.TensorProto.EXTERNAL
      weights_tensor.external_data.append(
        onnx.StringStringEntryProto(key="location", value="weights.bin")
      )

      inputs = [onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, (4,))]
      outputs = [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, (4,))]
      nodes = [onnx.helper.make_node('Add', ['input', 'weights'], ['output'])]

      graph = onnx.helper.make_graph(nodes, 'test_external', inputs, outputs, [weights_tensor])
      model = onnx.helper.make_model(graph)
      model_path = tmpdir_path / "model.onnx"
      with open(model_path, 'wb') as f:
        f.write(model.SerializeToString())

      runner = OnnxRunner(model_path)
      input_data = {'input': np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)}
      results = runner(input_data)

      expected = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
      np.testing.assert_array_equal(results['output'].numpy(), expected)

      self.assertTrue('weights' in runner.graph_values)
      np.testing.assert_array_equal(runner.graph_values['weights'].numpy(), weights_data)


SUPPORTED_DTYPES = [dt for dt, tdt in data_types.items() if is_dtype_supported(tdt)]
UNSUPPORTED_DTYPES = [dt for dt, tdt in data_types.items() if not is_dtype_supported(tdt)]


class TestOnnxRunnerDtypes(unittest.TestCase):
  def _test_dtype_context(self, onnx_dtype: int, context: str, expected_dtype: DType):
    if context == 'input':
      inputs = [onnx.helper.make_tensor_value_info('input', onnx_dtype, ())]
      outputs = [onnx.helper.make_tensor_value_info('output', onnx_dtype, ())]
      nodes = [onnx.helper.make_node('Identity', ['input'], ['output'])]
      runner, _ = run_onnx(nodes, inputs=inputs, outputs=outputs, input_data={'input': 1.0})
      self.assertEqual(runner.graph_inputs['input'].dtype, expected_dtype)
    elif context == 'initializer':
      init = onnx.helper.make_tensor('init', onnx_dtype, (2,), [1, 2])
      outputs = [onnx.helper.make_tensor_value_info('output', onnx_dtype, (2,))]
      nodes = [onnx.helper.make_node('Constant', [], ['output'], value=init)]
      runner, _ = run_onnx(nodes, outputs=outputs, initializers=[init])
      self.assertEqual(runner.graph_values['init'].dtype, expected_dtype)
    elif context == 'constant':
      value = onnx.helper.make_tensor('value', onnx_dtype, (2,), [1, 2])
      outputs = [onnx.helper.make_tensor_value_info('output', onnx_dtype, (2,))]
      nodes = [onnx.helper.make_node('Constant', [], ['output'], value=value)]
      runner, _ = run_onnx(nodes, outputs=outputs)
      self.assertEqual(runner.graph_nodes[0].opts['value'].dtype, expected_dtype)

  @given(onnx_dtype=st.sampled_from(SUPPORTED_DTYPES), context=st.sampled_from(['input', 'initializer', 'constant']))
  def test_supported_dtypes(self, onnx_dtype: int, context: str):
    expected = data_types[onnx_dtype]
    self._test_dtype_context(onnx_dtype, context, expected)

  @unittest.skipUnless(UNSUPPORTED_DTYPES, "No unsupported dtypes to test")
  @given( onnx_dtype=st.sampled_from(UNSUPPORTED_DTYPES), context=st.sampled_from(['input', 'initializer', 'constant']))
  def test_unsupported_dtypes(self, onnx_dtype: int, context: str):
    true_dtype = data_types[onnx_dtype]
    expected = true_dtype if context == "input" else dtypes.default_int if dtypes.is_int(true_dtype) else dtypes.default_float
    self._test_dtype_context(onnx_dtype, context, expected)


if __name__ == '__main__':
  unittest.main()