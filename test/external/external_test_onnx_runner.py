import unittest, onnx, tempfile, os
import numpy as np
from tinygrad.frontend.onnx import OnnxRunner

def helper_make_identity_model(inps: dict[str, np.ndarray], nm) -> onnx.ModelProto:
  onnx_inputs = [onnx.helper.make_tensor_value_info(name, onnx.helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inps.items()]
  onnx_outputs = [onnx.helper.make_tensor_value_info(name, onnx.helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inps.items()]
  nodes = [onnx.helper.make_node("Identity", list(inps), list(inps), domain="")]
  graph = onnx.helper.make_graph(nodes, f"test_{nm}", onnx_inputs, onnx_outputs)
  return onnx.helper.make_model(graph, producer_name=f"test_{nm}")

class TestOnnxRunnerModelLoading(unittest.TestCase):
  def setUp(self):
    self.x = np.array([1.0, 2.0], dtype=np.float32)
    self.model = helper_make_identity_model({"X": self.x}, "identity")

  def _run_identity_test(self, model):
    runner = OnnxRunner(model)
    result = runner({"X": self.x})
    np.testing.assert_equal(result["X"].numpy(), self.x)

  def test_model_load_from_file_path(self):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file: temp_path = tmp_file.name
    try:
      onnx.save(self.model, temp_path)
      self._run_identity_test(temp_path)
    finally: os.remove(temp_path)

  def test_model_load_from_file_like(self):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file: temp_path = tmp_file.name
    try:
      onnx.save(self.model, temp_path)
      with open(temp_path, "rb") as f: self._run_identity_test(f)
    finally: os.remove(temp_path)

  def test_model_load_from_raw_bytes(self): self._run_identity_test(self.model.SerializeToString())

if __name__ == '__main__':
  unittest.main()
