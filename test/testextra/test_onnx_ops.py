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
#
# if __name__ == "__main__":
#   unittest.main()