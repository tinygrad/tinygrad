import unittest, onnx, tempfile
from tinygrad import dtypes
from tinygrad.frontend.onnx import OnnxRunner, onnx_load
from tinygrad.device import is_dtype_supported
import numpy as np

class TestOnnxRunner(unittest.TestCase):
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