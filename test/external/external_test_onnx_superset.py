import numpy as np
from onnx import helper
from onnx.backend.test.case.test_case import TestCase
from onnx.defs import AI_ONNX_PREVIEW_TRAINING_DOMAIN

def create_testcase(op:str, name:str, inputs:dict[str, np.ndarray], outputs:dict[str, np.ndarray], domain='', rtol=1e-3, atol=1e-7, **opts):
  node = helper.make_node(op, list(inputs), outputs, name=f"test_{op}", domain=domain, **opts)
  inputs_info = [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inputs.items()]
  outputs_info = [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in outputs.items()]
  graph = helper.make_graph([node], name, inputs_info, outputs_info)
  model = helper.make_model(graph)
  return TestCase(name=name, model_name=name, model=model, data_sets=[(list(inputs.values()), list(outputs.values()))], kind="node",
                  rtol=rtol, atol=atol, url=None, model_dir=None)

def test_adam_large_t():
  from onnx.backend.test.case.node.adam import apply_adam
  op = "Adam"
  inputs = {
  "r": np.array(0.1, dtype=np.float32),
  "t": np.array(2, dtype=np.int64),
  "x": np.array([1.2, 2.8], dtype=np.float32),
  "g": np.array([-0.94, -2.5], dtype=np.float32),
  "v": np.array([1.7, 3.6], dtype=np.float32),
  "h": np.array([0.1, 0.1], dtype=np.float32)
  }
  outputs = ["x_new", "v_new", "h_new"]
  opts = { "alpha": 0.95, "beta": 0.1, "epsilon": 1e-7, "norm_coefficient": 0.001, "norm_coefficient_post": 0.0 }
  onnx_out = apply_adam(**inputs, **opts)
  outputs = dict([*list(zip(outputs, onnx_out))])
  # NOTE: fix x_new since np.sqrt turns float32 into float64
  outputs["x_new"] = outputs["x_new"].astype(np.float32)
  adam_test_case = create_testcase(op, "test_adam_internal", inputs, outputs, AI_ONNX_PREVIEW_TRAINING_DOMAIN, **opts)
  return adam_test_case

def test_adagrad_large_t():
  from onnx.backend.test.case.node.adagrad import apply_adagrad
  op = "Adagrad"
  inputs = {
    "r": np.array(0.1, dtype=np.float32),
    "t": np.array(2, dtype=np.int64),
    "x": np.array([1.0], dtype=np.float32),
    "g": np.array([-1.0], dtype=np.float32),
    "h": np.array([2.0], dtype=np.float32)
  }
  outputs = ["x_new", "h_new"]
  opts = { "norm_coefficient": 0.001, "epsilon": 1e-5, "decay_factor": 0.1 }
  onnx_out = apply_adagrad(**inputs, **opts)
  outputs = dict([*list(zip(outputs, onnx_out))])
  # NOTE: fix x_new since np.sqrt turns float32 into float64
  outputs["x_new"] = outputs["x_new"].astype(np.float32)
  adagrad_test_case = create_testcase(op, "test_adagrad_internal", inputs, outputs, AI_ONNX_PREVIEW_TRAINING_DOMAIN, **opts)
  return adagrad_test_case

def test_momentum_large_t():
  from onnx.backend.test.case.node.momentum import apply_momentum
  op = "Momentum"
  inputs = {
    "r": np.array(0.1, dtype=np.float32),
    "t": np.array(2, dtype=np.int64),
    "x": np.array([1.2, 2.8], dtype=np.float32),
    "g": np.array([-0.94, -2.5], dtype=np.float32),
    "v": np.array([1.7, 3.6], dtype=np.float32)
  }
  outputs = ["x_new", "v_new"]
  opts = { "norm_coefficient": 0.001, "alpha": 0.95, "beta": 0.1, "mode": "standard" }
  onnx_out = apply_momentum(**inputs, **{"norm_coefficient": 0.001, "alpha": 0.95, "beta": 0.1})
  outputs = dict([*list(zip(outputs, onnx_out))])
  momentum_test_case = create_testcase(op, "test_momentum_internal", inputs, outputs, AI_ONNX_PREVIEW_TRAINING_DOMAIN, **opts)
  return momentum_test_case

momentum_test_case = test_momentum_large_t()
adam_test_case = test_adam_large_t()
adagrad_test_case = test_adagrad_large_t()