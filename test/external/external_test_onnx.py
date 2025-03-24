# tests basic onnx properties

import unittest
import onnx
import numpy as np
from extra.onnx import OnnxRunner

def helper_make_identity_model(inps: dict[str, np.ndarray], nm) -> onnx.ModelProto:
  onnx_inputs = [onnx.helper.make_tensor_value_info(name, onnx.helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inps.items()]
  onnx_outputs = [onnx.helper.make_tensor_value_info(name, onnx.helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inps.items()]
  nodes = [onnx.helper.make_node("Identity", list(inps), list(inps), domain="")]
  graph = onnx.helper.make_graph(nodes, f"test_{nm}", onnx_inputs, onnx_outputs)
  return onnx.helper.make_model(graph, producer_name=f"test_{nm}")

class TestOnnxRunner(unittest.TestCase):
  def test_identity_op(self):
    # Build a model with a single input "X" and run identity op.
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    model = helper_make_identity_model({"X": x}, "identity")
    runner = OnnxRunner(model)
    result = runner({"X": x})
    np.testing.assert_array_almost_equal(result["X"].data(), x)

  def test_mismatched_input_shape(self):
    # Build a model with input "X" of shape (2, 3)
    x = np.ones((2, 3), dtype=np.float32)
    model = helper_make_identity_model({"X": x}, "mismatch")
    runner = OnnxRunner(model)
    # Provide an input with a mismatched shape (2, 4) and expect an error.
    x_bad = np.ones((2, 4), dtype=np.float32)
    with self.assertRaises(RuntimeError):
      runner({"X": x_bad})

  def test_identity_op_multiple_keys(self):
    # Build a model with two inputs "X" and "Y" and run identity op on both.
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([10.0, 20.0], dtype=np.float32)
    model = helper_make_identity_model({"X": x, "Y": y}, "multi")
    runner = OnnxRunner(model)
    result = runner({"X": x, "Y": y})
    np.testing.assert_array_almost_equal(result["X"].data(), x)
    np.testing.assert_array_almost_equal(result["Y"].data(), y)

if __name__ == '__main__':
  unittest.main()
