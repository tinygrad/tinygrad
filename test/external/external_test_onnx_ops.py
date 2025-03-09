# inputs, attributes, and outputs for tests are found here:
# https://github.com/onnx/onnx/blob/main/docs/Operators.md
# https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md

from typing import Any
import unittest, onnx, tempfile
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
    with tempfile.NamedTemporaryFile() as tmp:
      onnx.save(model, tmp.name)
      validate(tmp.name, inps, rtol, atol)

class TestMainOnnxOps(TestOnnxOps):
  DOMAIN = ""
  def test_reshape(self):
    inputs = {"in": np.arange(6, dtype=np.float32), "shape": np.array([2,3], dtype=np.int64)}
    attributes = {}
    outputs = ["out"]
    self.helper_test_single_op("Reshape", inputs, attributes, outputs)

  def test_conv(self):
    # test VALID auto_pad
    inputs = {
      "x": np.random.randn(1, 3, 384, 384).astype(np.float32),
      "w": np.random.randn(1152, 3, 14, 14).astype(np.float32),
      "b": np.random.randn(1152).astype(np.float32)
    }
    attributes = {'auto_pad': 'VALID', 'dilations': (1, 1), 'group': 1, 'kernel_shape': (14, 14), 'strides': (14, 14)}
    outputs = ["y"]
    self.helper_test_single_op("Conv", inputs, attributes, outputs, atol=1e-4)

  def test_gather(self):
    # test const negative indices
    inputs = {
      "input": np.random.randn(1, 3, 3).astype(np.float32),
      "indices": np.array(-2, dtype=np.int64),
    }
    attributes = {'axis': 1}
    outputs = ["y"]
    self.helper_test_single_op("Gather", inputs, attributes, outputs)

  def test_quantize_linear(self):
    test_cases = [
      {"test_case": "round_half_to_even", "qdtype": np.int8, "qzero_point": 0, "x": [-1.5, -0.5, 0.5, 1.5], "scale": 1.0},
      {"test_case": "round_to_even_before_add_zero_point", "qdtype": np.uint8, "qzero_point": 1, "x": [0.5, 1.5], "scale": 1.0},
    ]
    for case in test_cases:
      with self.subTest(test_case=case["test_case"]):
        inputs = {
          "x": np.array([case["x"]], dtype=np.float32),
          "y_scale": np.array(case["scale"], dtype=np.float32),
          "y_zero_point": np.array(case["qzero_point"], dtype=case["qdtype"])
        }
        self.helper_test_single_op("QuantizeLinear", inputs, {}, ["y"])

  def test_dynamic_quantize_linear(self):
    test_cases = [
      {"name": "round_half_to_even", "x": np.array([0, 0.5, 1.5, 255], dtype=np.float32)},
      {"name": "round_zero_point_half_down_to_even", "x": np.array([-1, 509], dtype=np.float32)},
      {"name": "round_zero_point_half_up_to_even", "x": np.array([-11, 499], dtype=np.float32)},
      # other tests from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-45
      {"name": "max_adjusted", "x": np.array([-1.0, -2.1, -1.3, -2.5, -3.34, -4.0], dtype=np.float32)},
      {"name": "min_adjusted", "x": np.array([1, 2.1, 1.3, 2.5, 3.34, 4.0, 1.5, 2.6, 3.9, 4.0, 3.0, 2.345], dtype=np.float32).reshape((3, 4))},
    ]
    for case in test_cases:
      with self.subTest(test_case=case["name"]):
        self.helper_test_single_op("DynamicQuantizeLinear", {"x": case["x"]}, {}, ["y", "y_scale", "y_zero_point"])

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
          self.helper_test_single_op("QLinearConv", inputs, attributes, outputs, atol=1) # occasionally inaccurate

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
        self.helper_test_single_op("QLinearMatMul", inputs, attributes, outputs)

    for name,val in (("round_half_down_to_even", 1), ("round_half_up_to_even", 3)):
      with self.subTest(test_case=name, val=val):
        inputs = {
          "A": np.array([val], dtype=np.int8),
          "A_scale": np.array(0.5, dtype=np.float32),
          "A_zero_point": np.array(0, dtype=np.int8),
          "B": np.array([1], dtype=np.int8),
          "B_scale": np.array(1, dtype=np.float32),
          "B_zero_point": np.array(0, dtype=np.int8),
          "Y_scale": np.array(1, dtype=np.float32),
          "Y_zero_point": np.array(0, dtype=np.int8)
        }
        attributes = {}
        outputs = ["Y"]
        self.helper_test_single_op("QLinearMatMul", inputs, attributes, outputs)

class TestContribOnnxOps(TestOnnxOps):
  DOMAIN = "com.microsoft"
  def test_attention(self):
    batch_size, seq_len, input_hidden_size = 2, 8, 256
    num_heads, head_size = 4, 64
    hidden_size = num_heads * head_size
    v_hidden_size = hidden_size

    # for mask_index
    right_padding_mask = np.random.randint(1, seq_len + 1, size=(batch_size,), dtype=np.int32)
    end_positions = np.random.randint(1, seq_len + 1, size=(batch_size,), dtype=np.int32)
    start_positions = np.array([np.random.randint(0, end) for end in end_positions], dtype=np.int32)
    left_padding_mask = np.concatenate([end_positions, start_positions])

    base_inps = {
      "input": np.random.randn(batch_size, seq_len, input_hidden_size).astype(np.float32),
      "weights": np.random.randn(input_hidden_size, hidden_size * 3).astype(np.float32),
      # bias is required in ORT (segfaults otherwise), eventhough docs says it's optional
      "bias": np.random.randn(hidden_size * 2 + v_hidden_size).astype(np.float32),
    }
    base_opts = {"num_heads": num_heads}

    test_cases = [
      ({}, {}),
      ({}, {"scale": 0.1}),
      ({}, {"scale": 1.0}),
      ({}, {"unidirectional": 1}),
      ({"mask_index": right_padding_mask}, {}),
      ({"mask_index": left_padding_mask}, {}),
      ({"mask_index": np.random.randint(0, seq_len, size=(batch_size, seq_len), dtype=np.int32)}, {"mask_filter_value": -5000.0}),
      ({"mask_index": np.random.randint(0, seq_len, size=(batch_size, seq_len, seq_len), dtype=np.int32)}, {"mask_filter_value": -np.inf}),
      # BUG: when `mask_index` is used with `unidirectional`, the first value must be True
      # otherwise this will trigger a different ORT behavior where start consecutive Falses will be turned True
      # e.g. mask_index = [[0, 0, 1, 0, 1, 1, 1, 1], [0, 0, 1, 0, 1, 1, 1, 1]]
      # will need mask[:, :, 0:1, 0:1] = True
      ({"mask_index": np.array([[1, 0, 1, 0, 1, 1, 1, 1], [1, 0, 1, 0, 1, 1, 1, 1]], dtype=np.int32)}, {"unidirectional": 1}),
      ({ "weights": np.random.randn(input_hidden_size, hidden_size + hidden_size + 128).astype(np.float32),
         "bias": np.random.randn(hidden_size + hidden_size + 128).astype(np.float32)},
       {"qkv_hidden_sizes": [hidden_size, hidden_size, 128]}),
      # TODO: past is not tested. ORT gives type error for input
    ]

    for i, (extra_inps, extra_opts) in enumerate(test_cases):
      with self.subTest(f"test_attention_{i}"):
        inps = {**base_inps, **extra_inps}
        opts = {**base_opts, **extra_opts}
        outputs = ["output", "present"] if "past" in inps else ["output"]
        self.helper_test_single_op("Attention", inps, opts, outputs, atol=1e-4)

  def test_skip_layer_normalization(self):
    shape = (2, 8, 32)
    for has_beta in [True, False]:
      for has_bias in [True, False]:
        with self.subTest(has_beta=has_beta, has_bias=has_bias):
          hidden_size = shape[-1]
          inputs = {
            "input": np.random.randn(*shape).astype(np.float32),
            "skip": np.random.randn(*shape).astype(np.float32),
            "gamma": np.random.randn(hidden_size).astype(np.float32),
          }
          if has_beta: inputs["beta"] = np.random.randn(hidden_size).astype(np.float32)
          if has_bias: inputs["bias"] = np.random.randn(hidden_size).astype(np.float32)
          attributes = {"epsilon": 1e-12}
          outputs = ["output", "mean", "inv_std_var", "input_skip_bias_sum"]
          self.helper_test_single_op("SkipLayerNormalization", inputs, attributes, outputs)

  def test_bias_gelu(self):
    shape = (2,3,4)
    inputs = {
      "A": np.random.randn(*shape).astype(np.float32),
      "B": np.random.randn(shape[-1]).astype(np.float32)
    }
    attributes = {}
    outputs = ["C"]
    self.helper_test_single_op("BiasGelu", inputs, attributes, outputs)

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

    with self.subTest(test_case="round_half_to_even"):
      inputs = {
        "A": np.array([1, 1, 1, 1], dtype=np.int8),
        "A_scale": np.array(1, dtype=np.float32),
        "A_zero_point": np.array(0, dtype=np.int8),
        "B": np.array([1, 5, -3, -7], dtype=np.int8),
        "B_scale": np.array(1, dtype=np.float32),
        "B_zero_point": np.array(0, dtype=np.int8),
        "C_scale": np.array(4, dtype=np.float32),
        "C_zero_point": np.array(0, dtype=np.int8)
      }
      attributes = {}
      outputs = ["C"]
      self.helper_test_single_op("QLinearAdd", inputs, attributes, outputs)

  def test_qlinear_mul(self):
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
        self.helper_test_single_op("QLinearMul", inputs, attributes, outputs)

    with self.subTest(test_case="round_half_to_even"):
      inputs = {
        "A": np.array([1, 1, 1, 1], dtype=np.int8),
        "A_scale": np.array(1, dtype=np.float32),
        "A_zero_point": np.array(0, dtype=np.int8),
        "B": np.array([2, 6, -2, -6], dtype=np.int8),
        "B_scale": np.array(1, dtype=np.float32),
        "B_zero_point": np.array(0, dtype=np.int8),
        "C_scale": np.array(4, dtype=np.float32),
        "C_zero_point": np.array(0, dtype=np.int8)
      }
      attributes = {}
      outputs = ["C"]
      self.helper_test_single_op("QLinearMul", inputs, attributes, outputs)

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
        self.helper_test_single_op("QLinearGlobalAveragePool", inputs, attributes, outputs)

if __name__ == "__main__":
  unittest.main()