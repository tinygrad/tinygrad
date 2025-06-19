import unittest, onnx, tempfile
from tinygrad import dtypes
from tinygrad.frontend.onnx import OnnxRunner, onnx_load
from tinygrad.device import is_dtype_supported
from extra.onnx import supported_dtypes, TensorDataType
import numpy as np

class TestOnnxRunner(unittest.TestCase):
  def _test_input_spec_dtype_parsing(self, onnx_tensor_dtype, tinygrad_dtype):
    input_tensor = onnx.helper.make_tensor_value_info('input', onnx_tensor_dtype, ())
    output_tensor = onnx.helper.make_tensor_value_info('output', onnx_tensor_dtype, ())
    node = onnx.helper.make_node('Identity', inputs=['input'], outputs=['output'])
    graph = onnx.helper.make_graph([node], 'identity_test', [input_tensor], [output_tensor])
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)
    with tempfile.NamedTemporaryFile() as tmp:
      onnx.save(model, tmp.name)
      tmp.flush()
      model = onnx_load(tmp.name)
      runner = OnnxRunner(model)
    assert len(runner.graph_inputs) == 1
    assert runner.graph_inputs['input'].dtype is tinygrad_dtype

  def test_input_spec_dtype_parsing(self):
    """ tests correct onnx_load parsing and dtype loading """
    if set(onnx.TensorProto.DataType.values()) != set(TensorDataType.__members__.values()):
      raise Exception("Official onnx datatypes and defined datatypes are out of sync, onnx vers changed, go update")
    for onnx_tensor_dtype in onnx.TensorProto.DataType.values():
      tensor_name = TensorDataType(onnx_tensor_dtype).name
      if onnx_tensor_dtype in supported_dtypes and is_dtype_supported(supported_dtypes[onnx_tensor_dtype]):
        with self.subTest(dtype=tensor_name):
          self._test_input_spec_dtype_parsing(onnx_tensor_dtype, supported_dtypes[onnx_tensor_dtype])
      else:
        print(f"{tensor_name} skipped.")

  @unittest.skipIf(is_dtype_supported(dtypes.int64), "only run test for unsupported dtypes")
  def test_dtype_fallback(self):
    node = onnx.helper.make_node('Gather', inputs=['data', 'indices'], outputs=['output'])
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([1], dtype=np.int64)
    data_tensor = onnx.helper.make_tensor_value_info('data', onnx.TensorProto.FLOAT, data.shape)
    indices_tensor = onnx.helper.make_tensor_value_info('indices', onnx.TensorProto.INT64, indices.shape)
    output_tensor = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 2])
    graph = onnx.helper.make_graph(
      [node],
      'gather_test',
      [data_tensor, indices_tensor],
      [output_tensor],
      [
        onnx.helper.make_tensor('data', onnx.TensorProto.FLOAT, data.shape, data.flatten()),
        onnx.helper.make_tensor('indices', onnx.TensorProto.INT64, indices.shape, indices.flatten())
      ]
    )
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)
    with tempfile.NamedTemporaryFile() as tmp:
      onnx.save(model, tmp.name)
      tmp.flush()
      model = onnx_load(tmp.name)
      runner = OnnxRunner(model)
    assert runner.graph_values['indices'].dtype == dtypes.default_int

if __name__ == '__main__':
  unittest.main()