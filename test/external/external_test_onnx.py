import unittest, onnx, tempfile, os
import numpy as np
from tinygrad.frontend.onnx import OnnxRunner

def helper_make_identity_model(inps: dict[str, np.ndarray], nm) -> onnx.ModelProto:
  onnx_inputs = [onnx.helper.make_tensor_value_info(name, onnx.helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inps.items()]
  onnx_outputs = [onnx.helper.make_tensor_value_info(name, onnx.helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inps.items()]
  nodes = [onnx.helper.make_node("Identity", list(inps), list(inps), domain="")]
  graph = onnx.helper.make_graph(nodes, f"test_{nm}", onnx_inputs, onnx_outputs)
  return onnx.helper.make_model(graph, producer_name=f"test_{nm}")

class TestOnnxRunner(unittest.TestCase):
  def test_model_load(self):
    def _test(model):
      runner = OnnxRunner(model)
      result = runner({"X": x})
      np.testing.assert_equal(result["X"].numpy(), x)
    x = np.array([1.0, 2.0], dtype=np.float32)
    model = helper_make_identity_model({"X": x}, "identity")

    # Load from a file path
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
      temp_path = tmp_file.name
    try:
      onnx.save(model, temp_path)
      _test(temp_path)
    finally:
      os.remove(temp_path)

    # Load from a file-like object
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
      temp_path = tmp_file.name
    try:
      onnx.save(model, temp_path)
      with open(temp_path, "rb") as f:
        _test(f)
    finally:
      os.remove(temp_path)

    # Load from raw bytes
    raw_bytes = model.SerializeToString()
    _test(raw_bytes)

    # Load from an onnx.ModelProto
    _test(model)

if __name__ == '__main__':
  unittest.main()
