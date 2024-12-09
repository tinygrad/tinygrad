import unittest, onnx
import numpy as np
from extra.onnx import get_run_onnx
from test.external.external_model_benchmark import assert_allclose
import onnxruntime as ort
ort_options = ort.SessionOptions()
ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_options.log_severity_level = 3  # no warnings
from onnx.helper import np_dtype_to_tensor_dtype, make_empty_tensor_value_info, make_tensor_value_info, make_graph, make_model

def helper_test_onnx_op(op:str, name:str, inputs:dict[str, np.ndarray], outputs:list[str], atol=1e-6, rtol=1e-3, **opts):
  # create model
  node = onnx.helper.make_node(op, list(inputs), outputs, name=f"test_{op}", **opts)
  inputs_info = [make_tensor_value_info(name, np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inputs.items()]
  outputs_info = [make_empty_tensor_value_info(name) for name in outputs]    # dummy outputs
  graph = make_graph([node], name, inputs_info, outputs_info)
  model = make_model(graph)

  # onnx
  ort_session = ort.InferenceSession(model.SerializeToString(), ort_options, ["CPUExecutionProvider"])
  onnx_out = ort_session.run(outputs, inputs)
  onnx_out = dict([*list(zip(outputs, onnx_out))])

  # tinygrad
  tinygrad_session = get_run_onnx(model)
  tiny_out = tinygrad_session(inputs)

  # verify
  assert_allclose(tiny_out, onnx_out, rtol, atol)

  # clean up
  del tinygrad_session, ort_session

class TestOnnxOps(unittest.TestCase):
  # src: https://github.com/onnx/onnx/blob/main/docs/Operators.md

  # example test
  def test_simple_Add(self):
    op = "Add"
    inputs = {"x": np.ones([3,3]), "y": np.ones([3,3])}
    outputs = ["sum"]
    helper_test_onnx_op(op, "test_simple_Add", inputs, outputs)

  # example test with weird opts
  def test_CumSum(self):
    op = "CumSum"
    inputs = {"tensor": np.arange(32).reshape(4,8), "dim": np.array(1)}
    outputs = ["cumsumed"]
    opts = {"exclusive":1, "reverse": 1}
    helper_test_onnx_op(op, "test_simple_Add", inputs, outputs, **opts)

if __name__ == "__main__":
  unittest.main()