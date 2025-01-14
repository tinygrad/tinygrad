# Spec (inputs, attributes, outputs) for tests found here:
# https://github.com/onnx/onnx/blob/main/docs/Operators.md
# https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md

from typing import Any
import unittest
import numpy as np

from tinygrad import Tensor
from tinygrad.helpers import getenv
from extra.onnx import get_run_onnx

import onnxruntime.backend
from onnx import helper

PRINT_TENSORS = getenv("PRINT_TENSORS", 0)
CONTRIB_OPERATORS = "com.microsoft"

def helper_test_op(inputs, model, atol=1e-6, rtol=1e-3):
  input_names = [inp.name for inp in model.graph.input]
  if not isinstance(inputs, list): inputs = [inputs]
  inp = dict(zip(input_names, inputs))

  rep = onnxruntime.backend.prepare(model.SerializeToString(), "CPU")
  ort_out = rep.run(inputs)

  tinygrad_runner = get_run_onnx(model)
  tinygrad_out = tinygrad_runner(inp)
  tinygrad_out = [out.numpy() if isinstance(out, Tensor) else out for out in tinygrad_out.values()]

  for tinygrad_val, ort_val in zip(tinygrad_out, ort_out):
    if PRINT_TENSORS: print(tinygrad_val, ort_val)
    assert tinygrad_val.dtype == ort_val.dtype, f"dtype mismatch: tinygrad={tinygrad_val.dtype} | onnxruntime={ort_val.dtype}"
    np.testing.assert_allclose(tinygrad_val, ort_val, rtol=rtol, atol=atol)

def helper_test_single_op(op:str, inps:dict[str, np.ndarray], opt:dict[str, Any],
                          outs:dict[str, tuple[list[int], np.dtype]], domain=None, atol=1e-6, rtol=1e-3, tag=""):
  onnx_inputs = [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inps.items()]
  onnx_outputs = [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), shape)
                  for name, (shape, dtype) in outs.items()]
  nodes = [helper.make_node(op, list(inps.keys()), list(outs), domain=domain, **opt)]
  graph = helper.make_graph(nodes, f"test_{op.lower()}{tag}", onnx_inputs, onnx_outputs)
  model = helper.make_model(graph, producer_name=f"test_{op.lower()}{tag}")
  helper_test_op(list(inps.values()), model, atol, rtol)

class TestOnnxOps(unittest.TestCase):
  def test_reshape(self):
    inputs = {"in": np.arange(6, dtype=np.float32), "shape": np.array([2,3], dtype=np.int64)}
    attributes = {}
    outputs = {"out": ([2,3], np.float32)}
    helper_test_single_op("Reshape", inputs, attributes, outputs)

class TestOnnxQuantizedOps(unittest.TestCase):
  def test_qlinear_conv(self):
    for dtype, zero_point in [(np.uint8, 128), (np.int8, 0)]:
      with self.subTest(dtype=dtype, zero_point=zero_point):
        dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max+1
        inputs = {
          "x": np.random.randint(dtype_min, dtype_max, [1, 3, 224, 224], dtype=dtype),
          "x_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "x_zero_point": np.array(zero_point, dtype=dtype),
          "w": np.random.randint(dtype_min, dtype_max, [32, 3, 3, 3], dtype=dtype),
          "w_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "w_zero_point": np.array(zero_point, dtype=dtype),
          "y_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "y_zero_point": np.array(zero_point, dtype=dtype),
          "b": np.random.randint(-10000, 10000, [32], dtype=np.int32)
        }
        attributes = {'auto_pad': 'NOTSET', 'dilations': (1, 1), 'group': 1, 'kernel_shape': (3, 3), 'pads': (1, 1, 1, 1), 'strides': (2, 2)}
        outputs = {"out": ([1, 32, 112, 112], dtype)}
        helper_test_single_op("QLinearConv", inputs, attributes, outputs, atol=1)

  def test_qlinear_matmul(self):
    for dtype, zero_point in [(np.uint8, 128), (np.int8, 0)]:
      with self.subTest(dtype=dtype, zero_point=zero_point):
        dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max+1
        inputs = {
          "A": np.random.randint(dtype_min, dtype_max, [10, 10], dtype=dtype),
          "A_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "A_zero_point": np.array(zero_point, dtype=dtype),
          "B": np.random.randint(dtype_min, dtype_max, [10, 10], dtype=dtype),
          "B_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "B_zero_point": np.array(zero_point, dtype=dtype),
          "Y_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "Y_zero_point": np.array(zero_point, dtype=dtype)
        }
        attributes = {}
        outputs = {"Y": ([10,10], dtype)}
        helper_test_single_op("QLinearMatMul", inputs, attributes, outputs, atol=1)

  def test_qlinear_add(self):
    for dtype, zero_point in [(np.uint8, 128), (np.int8, 0)]:
      with self.subTest(dtype=dtype, zero_point=zero_point):
        dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max+1
        inputs = {
          "A": np.random.randint(dtype_min, dtype_max, [10, 10], dtype=dtype),
          "A_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "A_zero_point": np.array(zero_point, dtype=dtype),
          "B": np.random.randint(dtype_min, dtype_max, [10, 10], dtype=dtype),
          "B_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "B_zero_point": np.array(zero_point, dtype=dtype),
          "C_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "C_zero_point": np.array(zero_point, dtype=dtype)
        }
        attributes = {}
        outputs = {"C": ([10,10], dtype)}
        helper_test_single_op("QLinearAdd", inputs, attributes, outputs, domain=CONTRIB_OPERATORS)

  # TODO: test channels_last
  def test_qlinear_global_average_pool(self):
    for dtype, zero_point in [(np.uint8, 128), (np.int8, 0)]:
      with self.subTest(dtype=dtype, zero_point=zero_point):
        dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max+1
        inputs = {
          "X": np.random.randint(dtype_min, dtype_max, [1, 3, 32, 32], dtype=dtype),
          "x_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "x_zero_point": np.array(zero_point, dtype=dtype),
          "y_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
          "y_zero_point": np.array(zero_point, dtype=dtype)
        }
        attributes = {"channels_last": 0}
        outputs = {"Y": ([1,3,1,1], dtype)}
        helper_test_single_op("QLinearGlobalAveragePool", inputs, attributes, outputs, domain=CONTRIB_OPERATORS, atol=1)

  def test_qgemm(self):
    for dtype, zero_point in [(np.uint8, 128), (np.int8, 0)]:
      for alpha in [0.5, 1.0, 2.0]:
        for transA in [0, 1]:
          with self.subTest(dtype=dtype, zero_point=zero_point, alpha=alpha, transA=transA):
            dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max+1
            inputs = {
              "A": np.random.randint(dtype_min, dtype_max, [32, 32], dtype=dtype),
              "a_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
              "a_zero_point": np.array(zero_point, dtype=dtype),
              "B": np.random.randint(dtype_min, dtype_max, [32, 32], dtype=dtype),
              "b_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
              "b_zero_point": np.array(zero_point, dtype=dtype),
              "C": np.random.randint(-16256, 16256, [32, 32], dtype=np.int32),
            }
            attributes = {'alpha': alpha, 'transA': transA, 'transB': 0}
            outputs = {"Y": ([32,32], np.float32)}
            helper_test_single_op("QGemm", inputs, attributes, outputs, domain=CONTRIB_OPERATORS, atol=1)

            inputs = {**inputs,
                      "y_scale": np.array(np.random.uniform(0.01, 0.1), dtype=np.float32),
                      "y_zero_point": np.array(zero_point, dtype=dtype)}
            outputs = {"Y": ([32,32], dtype)}
            helper_test_single_op("QGemm", inputs, attributes, outputs, domain=CONTRIB_OPERATORS, atol=1)

if __name__ == "__main__":
  unittest.main()