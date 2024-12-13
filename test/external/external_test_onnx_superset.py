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

# ground truth either comes from `onnx.backend.test.case.node` or running ort and pasting the result in
def test_adam_large_t():
  from onnx.backend.test.case.node.adam import apply_adam
  op = "Adam"
  inputs = {"r": np.array(0.1, dtype=np.float32), "t": np.array(2, dtype=np.int64), "x": np.array([1.2, 2.8], dtype=np.float32),
  "g": np.array([-0.94, -2.5], dtype=np.float32), "v": np.array([1.7, 3.6], dtype=np.float32), "h": np.array([0.1, 0.1], dtype=np.float32)}
  outputs = ["x_new", "v_new", "h_new"]
  opts = {"alpha": 0.95, "beta": 0.1, "epsilon": 1e-7, "norm_coefficient": 0.001, "norm_coefficient_post": 0.0}
  onnx_out = apply_adam(**inputs, **opts)
  outputs = dict([*list(zip(outputs, onnx_out))])
  # NOTE: fix x_new since np.sqrt turns float32 into float64
  outputs["x_new"] = outputs["x_new"].astype(np.float32)
  adam_test_case = create_testcase(op, "test_adam_internal", inputs, outputs, AI_ONNX_PREVIEW_TRAINING_DOMAIN, **opts)
  return adam_test_case

def test_adam_multiple():
  from onnx.backend.test.case.node.adam import apply_adam
  op = "Adam"
  inputs = {"r": np.array(0.1, dtype=np.float32), "t": np.array(0, dtype=np.int64), "x1": np.array([1.0], dtype=np.float32),
    "x2": np.array([1.0, 2.0], dtype=np.float32), "g1": np.array([-1.0], dtype=np.float32), "g2": np.array([-1.0, -3.0], dtype=np.float32),
    "v1": np.array([2.0], dtype=np.float32), "v2": np.array([4.0, 1.0], dtype=np.float32), "h1": np.array([0.5], dtype=np.float32),
    "h2": np.array([1.0, 10.0], dtype=np.float32)}
  outputs = ["x1_new", "x2_new", "v1_new", "v2_new", "h1_new", "h2_new"]
  opts = {"alpha": 0.95, "beta": 0.85, "epsilon": 1e-2, "norm_coefficient": 0.001, "norm_coefficient_post": 0.0}

  x1_new, v1_new, h1_new = apply_adam(inputs["r"], inputs["t"], inputs["x1"], inputs["g1"], inputs["v1"], inputs["h1"],
                                     opts["norm_coefficient"], opts["norm_coefficient_post"], opts["alpha"], opts["beta"], opts["epsilon"])
  x2_new, v2_new, h2_new = apply_adam(inputs["r"], inputs["t"], inputs["x2"], inputs["g2"], inputs["v2"], inputs["h2"],
                                     opts["norm_coefficient"], opts["norm_coefficient_post"], opts["alpha"], opts["beta"], opts["epsilon"])

  outputs = dict(zip(outputs, [x1_new, x2_new, v1_new, v2_new, h1_new, h2_new]))
  return create_testcase(op, "test_adam_multiple", inputs, outputs, AI_ONNX_PREVIEW_TRAINING_DOMAIN, **opts)

def test_adagrad_large_t():
  from onnx.backend.test.case.node.adagrad import apply_adagrad
  op = "Adagrad"
  inputs = {"r": np.array(0.1, dtype=np.float32), "t": np.array(2, dtype=np.int64), "x": np.array([1.0], dtype=np.float32),
    "g": np.array([-1.0], dtype=np.float32), "h": np.array([2.0], dtype=np.float32)}
  outputs = ["x_new", "h_new"]
  opts = {"norm_coefficient": 0.001, "epsilon": 1e-5, "decay_factor": 0.1}
  onnx_out = apply_adagrad(**inputs, **opts)
  outputs = dict([*list(zip(outputs, onnx_out))])
  # NOTE: fix x_new since np.sqrt turns float32 into float64
  outputs["x_new"] = outputs["x_new"].astype(np.float32)
  return create_testcase(op, "test_adagrad_internal", inputs, outputs, AI_ONNX_PREVIEW_TRAINING_DOMAIN, **opts)

def test_momentum_large_t():
  from onnx.backend.test.case.node.momentum import apply_momentum
  op = "Momentum"
  inputs = {"r": np.array(0.1, dtype=np.float32), "t": np.array(2, dtype=np.int64), "x": np.array([1.2, 2.8], dtype=np.float32),
    "g": np.array([-0.94, -2.5], dtype=np.float32), "v": np.array([1.7, 3.6], dtype=np.float32)}
  outputs = ["x_new", "v_new"]
  opts = {"norm_coefficient": 0.001, "alpha": 0.95, "beta": 0.1, "mode": "standard"}
  onnx_out = apply_momentum(**inputs, **{"norm_coefficient": 0.001, "alpha": 0.95, "beta": 0.1})
  outputs = dict([*list(zip(outputs, onnx_out))])
  return create_testcase(op, "test_momentum_internal", inputs, outputs, AI_ONNX_PREVIEW_TRAINING_DOMAIN, **opts)


# import unittest
# import numpy as np
# from extra.onnx import get_run_onnx
# from test.external.external_model_benchmark import assert_allclose
# import onnxruntime as ort
# ort_options = ort.SessionOptions()
# ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# ort_options.log_severity_level = 3  # no warnings
# from onnx import helper
#
# def run_tinygrad(model, inputs):
#   tinygrad_session = get_run_onnx(model)
#   tinygrad_out = tinygrad_session(inputs)
#   del tinygrad_session
#   return tinygrad_out
#
# def run_ort(model, inputs, outputs):
#   ort_session = ort.InferenceSession(model.SerializeToString(), ort_options, ["CPUExecutionProvider"])
#   onnx_out = ort_session.run(outputs, inputs)
#   onnx_out = dict([*list(zip(outputs, onnx_out))])
#   del ort_session
#   return onnx_out
#
# def create_model(op:str, name:str, inputs:dict[str, np.ndarray], outputs:list[str], **opts):
#   node = helper.make_node(op, list(inputs), outputs, name=f"test_{op}", **opts)
#   inputs_info = [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inputs.items()]
#   outputs_info = [helper.make_empty_tensor_value_info(name) for name in outputs]    # dummy outputs
#   graph = helper.make_graph([node], name, inputs_info, outputs_info)
#   model = helper.make_model(graph)
#   return model
#
# def helper_test_vs_ort(op:str, name:str, inputs:dict[str, np.ndarray], outputs:list[str], atol=1e-6, rtol=1e-3, **opts):
#   model = create_model(op, name, inputs, outputs, **opts)
#   ort_out = run_ort(model, inputs, outputs)
#   tiny_out = run_tinygrad(model, inputs)
#   assert_allclose(tiny_out, ort_out, rtol, atol)