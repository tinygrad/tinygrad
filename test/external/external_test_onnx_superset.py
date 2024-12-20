import numpy as np
from onnx import helper
from onnx.backend.test.case.test_case import TestCase
from onnx.defs import AI_ONNX_PREVIEW_TRAINING_DOMAIN

# ******* HELPERS *******

def create_testcase(op:str, name:str, inputs:dict[str, np.ndarray], outputs:dict[str, np.ndarray], domain='', rtol=1e-3, atol=1e-7, **opts):
  node = helper.make_node(op, list(inputs), outputs, name=f"test_{op}", domain=domain, **opts)
  def parse_info(name, val):
    if isinstance(val, np.ndarray): return helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(val.dtype), val.shape)
    elif val is None: return helper.make_empty_tensor_value_info(name)
    else: raise RuntimeError(f"{val=}")
  inputs_info = [parse_info(name, val) for name, val in inputs.items()]
  outputs_info = [parse_info(name, val) for name, val in outputs.items()]
  graph = helper.make_graph([node], name, inputs_info, outputs_info)
  model = helper.make_model(graph)
  return TestCase(name=name, model_name=name, model=model, data_sets=[(list(inputs.values()), list(outputs.values()))], kind="node",
                  rtol=rtol, atol=atol, url=None, model_dir=None)

# ******* TESTS *******

# ground truth either comes from `onnx.backend.test.case.node` or running ort and copy-pasting the result in as output

def test_adam_large_training_iteration():
  from onnx.backend.test.case.node.adam import apply_adam
  op = "Adam"
  inputs = {"r": np.array(0.1, dtype=np.float32), "t": np.array(2, dtype=np.int64), "x": np.array([1.2, 2.8], dtype=np.float32),
  "g": np.array([-0.94, -2.5], dtype=np.float32), "v": np.array([1.7, 3.6], dtype=np.float32), "h": np.array([0.1, 0.1], dtype=np.float32)}
  outputs = ["x_new", "v_new", "h_new"]
  opts = {"alpha": 0.95, "beta": 0.1, "epsilon": 1e-7, "norm_coefficient": 0.001, "norm_coefficient_post": 0.0}
  onnx_out = apply_adam(**inputs, **opts)
  outputs = dict(zip(outputs, onnx_out))
  # NOTE: fix x_new since np.sqrt turns float32 into float64
  outputs["x_new"] = outputs["x_new"].astype(np.float32)
  return create_testcase(op, "test_adam_internal", inputs, outputs, AI_ONNX_PREVIEW_TRAINING_DOMAIN, **opts)

def test_adam_multiple():
  from onnx.backend.test.case.node.adam import apply_adam
  op = "Adam"
  inputs = {"r": np.array(0.1, dtype=np.float32), "t": np.array(0, dtype=np.int64), "x1": np.array([1.0], dtype=np.float32),
    "x2": np.array([1.0, 2.0], dtype=np.float32), "g1": np.array([-1.0], dtype=np.float32), "g2": np.array([-1.0, -3.0], dtype=np.float32),
    "v1": np.array([2.0], dtype=np.float32), "v2": np.array([4.0, 1.0], dtype=np.float32), "h1": np.array([0.5], dtype=np.float32),
    "h2": np.array([1.0, 10.0], dtype=np.float32)}
  outputs = ["x1_new", "x2_new", "v1_new", "v2_new", "h1_new", "h2_new"]
  opts = {"alpha": 0.95, "beta": 0.85, "epsilon": 1e-2, "norm_coefficient": 0.001, "norm_coefficient_post": 0.0}
  x1_new, v1_new, h1_new = apply_adam(inputs["r"], inputs["t"], inputs["x1"], inputs["g1"], inputs["v1"], inputs["h1"], **opts)
  x2_new, v2_new, h2_new = apply_adam(inputs["r"], inputs["t"], inputs["x2"], inputs["g2"], inputs["v2"], inputs["h2"], **opts)
  outputs = dict(zip(outputs, [x1_new, x2_new, v1_new, v2_new, h1_new, h2_new]))
  return create_testcase(op, "test_adam_multiple", inputs, outputs, AI_ONNX_PREVIEW_TRAINING_DOMAIN, **opts)

def test_adagrad_large_training_iteration():
  from onnx.backend.test.case.node.adagrad import apply_adagrad
  op = "Adagrad"
  inputs = {"r": np.array(0.1, dtype=np.float32), "t": np.array(2, dtype=np.int64), "x": np.array([1.0], dtype=np.float32),
    "g": np.array([-1.0], dtype=np.float32), "h": np.array([2.0], dtype=np.float32)}
  outputs = ["x_new", "h_new"]
  opts = {"norm_coefficient": 0.001, "epsilon": 1e-5, "decay_factor": 0.1}
  onnx_out = apply_adagrad(**inputs, **opts)
  outputs = dict(zip(outputs, onnx_out))
  # NOTE: fix x_new since np.sqrt turns float32 into float64
  outputs["x_new"] = outputs["x_new"].astype(np.float32)
  return create_testcase(op, "test_adagrad_internal", inputs, outputs, AI_ONNX_PREVIEW_TRAINING_DOMAIN, **opts)

def _test_momentum(mode):
  if mode == "standard":
    from onnx.backend.test.case.node.momentum import apply_momentum
  elif mode == "nesterov":
    from onnx.backend.test.case.node.momentum import apply_nesterov as apply_momentum
  op = "Momentum"
  inputs = {"r": np.array(0.1, dtype=np.float32), "t": np.array(2, dtype=np.int64), "x": np.array([1.2, 2.8], dtype=np.float32),
    "g": np.array([-0.94, -2.5], dtype=np.float32), "v": np.array([1.7, 3.6], dtype=np.float32)}
  outputs = ["x_new", "v_new"]
  opts = {"norm_coefficient": 0.001, "alpha": 0.95, "beta": 0.1, "mode": "standard"}
  onnx_out = apply_momentum(**inputs, **{"norm_coefficient": 0.001, "alpha": 0.95, "beta": 0.1})
  outputs = dict(zip(outputs, onnx_out))
  return create_testcase(op, f"test_{mode}_momentum_internal", inputs, outputs, AI_ONNX_PREVIEW_TRAINING_DOMAIN, **opts)

def test_momentum_large_training_iteration(): return _test_momentum("standard")
def test_nesterov_momentum_large_training_iteration(): return _test_momentum("nesterov")

def test_max_unpool_pads():
  op = "MaxUnpool"
  inputs = {
      "xT": np.array([[[[1, 3],
                        [9, 11]]]], dtype=np.float32),
      "xI": np.array([[[[1, 3],
                        [9, 11]]]], dtype=np.int64)
  }
  opts = {"kernel_shape": [2, 2], "strides": [2, 2], "pads": [1, 0, 0, 0]}
  outputs = {"y": np.array([[[[0.,1.,0.,3.], [0.,0.,0.,0.], [0.,9.,0.,11.]]]], dtype=np.float32)}
  return create_testcase(op, "test_maxunpool_pads_internal", inputs, outputs, **opts)

def test_gathernd_large_batch_dims():
  from onnx.backend.test.case.node.gathernd import gather_nd_impl
  op = "GatherND"
  inputs = {
    "data": np.array([[[[[0,1],[2,3]],[[4,5],[6,7]]],[[[8,9],[10,11]],[[12,13],[14,15]]]]], dtype=np.float32),
    "indices": np.array([[[[[0]],[[1]]],[[[1]],[[0]]]]], dtype=np.int64),
  }
  opts = {"batch_dims": 3}
  outputs = ["out"]
  onnx_out = gather_nd_impl(**inputs, **opts)
  outputs = dict(zip(outputs, [onnx_out]))
  return create_testcase(op=op, name="test_gathernd_large_batch_dims_interal", inputs=inputs, outputs=outputs, **opts)

def test_gathernd_large_batch_dims_multiple_indices():
  from onnx.backend.test.case.node.gathernd import gather_nd_impl
  op = "GatherND"
  inputs = {
    "data": np.array([[[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]]],[[[12,13,14],[15,16,17]],[[18,19,20],[21,22,23]]]], dtype=np.float32),
    "indices": np.array([[[[0,1]],[[1,0]]],[[[1,1]],[[0,0]]]], dtype=np.int64)
  }
  opts = {"batch_dims": 2}
  outputs = ["out"]
  onnx_out = gather_nd_impl(**inputs, **opts)
  outputs = dict(zip(outputs, [onnx_out]))
  return create_testcase(op=op, name="test_gathernd_large_batch_dims_multiple_indices_internal", inputs=inputs, outputs=outputs, **opts)

def test_scatternd_duplicate_indices_none_reduction():
  from onnx.backend.test.case.node.scatternd import scatter_nd_impl
  op = "ScatterND"
  inputs = {
    "data": np.array([0,1,2]),
    "indices": np.array([[1],[1]]),
    "updates": np.array([99,100])
  }
  outputs = ["out"]
  onnx_out = scatter_nd_impl(**inputs)
  outputs = dict(zip(outputs, [onnx_out]))
  return create_testcase(op=op, name="test_scatternd_duplicate_indices_none_reduction_internal", inputs=inputs, outputs=outputs)

TEST_CASES = [
  test_adam_large_training_iteration(),
  test_adagrad_large_training_iteration(),
  test_momentum_large_training_iteration(),
  test_gathernd_large_batch_dims(),
  test_gathernd_large_batch_dims_multiple_indices(),
  test_scatternd_duplicate_indices_none_reduction(),
  # TODO: fix
  # test_nesterov_momentum_large_training_iteration(),
  # test_max_unpool_pads(),
]

if __name__ == "__main__":
  # running __main__ is only for running run_ort for getting the outputs
  # outputs are then added into test functions and then imported through TEST_CASES
  import onnxruntime as ort
  ort_options = ort.SessionOptions()
  ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  ort_options.log_severity_level = 3  # no warnings

  def run_ort(op, inputs:dict[str, np.ndarray], outputs:list[str], domain:str="", **opts):
    node = helper.make_node(op, list(inputs), outputs, name=f"test_{op}", domain=domain, **opts)
    inputs_info = [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape) for name, arr in inputs.items()]
    outputs_info = [helper.make_empty_tensor_value_info(name) for name in outputs]    # dummy outputs
    graph = helper.make_graph([node], "dummy_name", inputs_info, outputs_info)
    model = helper.make_model(graph)
    ort_session = ort.InferenceSession(model.SerializeToString(), ort_options, ["CPUExecutionProvider"])
    onnx_out = ort_session.run(outputs, inputs)
    onnx_out = dict([*list(zip(outputs, onnx_out))])
    del ort_session
    return onnx_out

  # # com.microsoft Ops tests
  # def test_skiplayernormalization():
  #   op = "SkipLayerNormalization"
  #   inputs = {
  #     "x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
  #     "skip": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
  #     "gamma": np.array([0.5, 0.5, 0.5], dtype=np.float32),
  #     "beta": np.array([0.1, 0.1, 0.1], dtype=np.float32),
  #     "bias": np.array([0.1, 0.1, 0.1], dtype=np.float32)
  #   }
  #   outputs = run_ort(op, inputs, ["output", "mean", "inv_std_var", "skip_out"], "com.microsoft")
  #   return create_testcase(op, "test_skiplayernormalization", inputs, outputs, "com.microsoft")
  #
  # def test_fastgelu():
  #   op = "FastGelu"
  #   inputs = {
  #     "x": np.array([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]], dtype=np.float32),
  #     "bias": np.array([0.1, 0.2, 0.3], dtype=np.float32)
  #   }
  #   outputs = run_ort(op, inputs, ["output"], domain="com.microsoft")
  #   return create_testcase(op, "test_fastgelu", inputs, outputs, domain="com.microsoft")
  #
  # def test_embededlayernormalization():
  #   op = "EmbedLayerNormalization"
  #   inputs = {
  #     "input_ids": np.array([[1, 2, 3, 4]], dtype=np.int32),
  #     "segment_ids": np.array([[0, 0, 1, 1]], dtype=np.int32),
  #     "word_embedding": np.random.randn(10, 3).astype(np.float32),
  #     "position_embedding": np.random.randn(4, 3).astype(np.float32),
  #     "segment_embedding": np.random.randn(2, 3).astype(np.float32),
  #     "gamma": np.array([0.5, 0.5, 0.5], dtype=np.float32),
  #     "beta": np.array([0.1, 0.1, 0.1], dtype=np.float32)
  #   }
  #   outputs = run_ort(op, inputs, ["output", "mask_index", "embedding_sum"], domain="com.microsoft")
  #   return create_testcase(op, "test_embededlayernormalization", inputs, outputs, domain="com.microsoft")
  #
  # def test_attention():
  #   op = "Attention"
  #   inputs = {
  #     "x": np.random.randn(2, 4, 12).astype(np.float32),
  #     "weights": np.random.randn(12, 36).astype(np.float32),
  #     "bias": np.random.randn(36).astype(np.float32)
  #   }
  #   opts = {"num_heads": 3, "qkv_hidden_sizes": [12, 12, 12], "unidirectional": False}
  #   outputs = run_ort(op, inputs, ["output", "present"], domain="com.microsoft", **opts)
  #   return create_testcase(op, "test_attention", inputs, outputs, domain="com.microsoft", **opts)