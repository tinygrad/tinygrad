# inputs, attributes, and outputs for tests are found here:
# https://github.com/onnx/onnx/blob/main/docs/Operators.md
# https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md

from typing import Any
import unittest, os, onnx
import numpy as np
from extra.onnx_helpers import validate

class TestOnnxOps(unittest.TestCase):
  DOMAIN = None
  def helper_test_single_op(self, op:str, inps:dict[str, np.ndarray], opts:dict[str, Any], outs:list[str], rtol=1e-3, atol=1e-6):
    onnx_inputs = [onnx.helper.make_tensor_value_info(name, onnx.helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inps.items()]
    onnx_outputs = [onnx.helper.make_empty_tensor_value_info(name) for name in outs]
    nodes = [onnx.helper.make_node(op, list(inps), list(outs), domain=self.DOMAIN, **opts)]
    graph = onnx.helper.make_graph(nodes, f"test_{op.lower()}", onnx_inputs, onnx_outputs)
    model = onnx.helper.make_model(graph, producer_name=f"test_{op.lower()}")
    file_name = f"test_{op.lower()}.onnx"
    onnx.save(model, file_name)
    validate(file_name, inps, rtol, atol)
    os.remove(file_name)

class TestMainOps(TestOnnxOps):
  DOMAIN = ""
  def test_reshape(self):
    inputs = {"in": np.arange(6, dtype=np.float32), "shape": np.array([2,3], dtype=np.int64)}
    attributes = {}
    outputs = ["out"]
    self.helper_test_single_op("Reshape", inputs, attributes, outputs)

  def test_qlinear_conv(self):
    for dtype, zero_point in [(np.uint8, 128), (np.int8, 0)]:
      for b in (np.ones([32], dtype=np.int32), np.zeros([32], dtype=np.int32)):
        with self.subTest(dtype=dtype, zero_point=zero_point):
          dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max
          inputs = {
            "x": np.random.randint(dtype_min, dtype_max + 1, [1, 3, 224, 224], dtype=dtype),
            "x_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
            "x_zero_point": np.array(zero_point, dtype=dtype),
            "w": np.random.randint(dtype_min, dtype_max + 1, [32, 3, 3, 3], dtype=dtype),
            "w_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
            "w_zero_point": np.array(zero_point, dtype=dtype),
            "y_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
            "y_zero_point": np.array(zero_point, dtype=dtype),
            "b": b
          }
          attributes = {'auto_pad': 'NOTSET', 'dilations': (1, 1), 'group': 1, 'kernel_shape': (3, 3), 'pads': (1, 1, 1, 1), 'strides': (2, 2)}
          outputs = ["out"]
          self.helper_test_single_op("QLinearConv", inputs, attributes, outputs, atol=1)

  def test_qlinear_matmul(self):
    for dtype, zero_point in [(np.uint8, 128), (np.int8, 0)]:
      with self.subTest(dtype=dtype, zero_point=zero_point):
        dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max
        inputs = {
          "A": np.random.randint(dtype_min, dtype_max + 1, [10, 10], dtype=dtype),
          "A_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "A_zero_point": np.array(zero_point, dtype=dtype),
          "B": np.random.randint(dtype_min, dtype_max + 1, [10, 10], dtype=dtype),
          "B_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "B_zero_point": np.array(zero_point, dtype=dtype),
          "Y_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "Y_zero_point": np.array(zero_point, dtype=dtype)
        }
        attributes = {}
        outputs = ["Y"]
        self.helper_test_single_op("QLinearMatMul", inputs, attributes, outputs, atol=1)

class TestContribOps(TestOnnxOps):
  DOMAIN = "com.microsoft"
  def test_attention(self):
    ...
  def test_qlinear_add(self):
    for dtype, zero_point in [(np.uint8, 128), (np.int8, 0)]:
      with self.subTest(dtype=dtype, zero_point=zero_point):
        dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max
        inputs = {
          "A": np.random.randint(dtype_min, dtype_max + 1, [10, 10], dtype=dtype),
          "A_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "A_zero_point": np.array(zero_point, dtype=dtype),
          "B": np.random.randint(dtype_min, dtype_max + 1, [10, 10], dtype=dtype),
          "B_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "B_zero_point": np.array(zero_point, dtype=dtype),
          "C_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "C_zero_point": np.array(zero_point, dtype=dtype)
        }
        attributes = {}
        outputs = ["C"]
        self.helper_test_single_op("QLinearAdd", inputs, attributes, outputs)

  # TODO: test channels_last
  def test_qlinear_global_average_pool(self):
    for dtype, zero_point in [(np.uint8, 128), (np.int8, 0)]:
      with self.subTest(dtype=dtype, zero_point=zero_point):
        dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max
        inputs = {
          "X": np.random.randint(dtype_min, dtype_max + 1, [1, 3, 32, 32], dtype=dtype),
          "x_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "x_zero_point": np.array(zero_point, dtype=dtype),
          "y_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "y_zero_point": np.array(zero_point, dtype=dtype)
        }
        attributes = {"channels_last": 0}
        outputs = ["C"]
        self.helper_test_single_op("QLinearGlobalAveragePool", inputs, attributes, outputs, atol=1)

if __name__ == "__main__":
  unittest.main()