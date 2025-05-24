import unittest
import onnx
from onnx import TensorProto, helper, checker
import numpy as np
import os
from extra.onnx_helpers import validate
from tinygrad import Context

def create_onnx_model_half_precision_sum_reduce(input_shape, model_path="test_sum_reduce_half.onnx"):
  X = helper.make_tensor_value_info('input', TensorProto.FLOAT16, input_shape)

  sin_node = helper.make_node('Sin', ['input'], ['sin_out'])

  const1_val_fp16 = np.array(0.5, dtype=np.float16)
  const1_tensor = helper.make_tensor('const1_fp16', TensorProto.FLOAT16, [], const1_val_fp16.tobytes(), raw=True)
  mul_node = helper.make_node('Mul', ['sin_out', 'const1_fp16'], ['mul_out_fp16'])

  reduce_axes_tensor = helper.make_tensor('reduce_axes', TensorProto.INT64, [1], [1])
  reduce_sum_node = helper.make_node(
    'ReduceSum',
    ['mul_out_fp16', 'reduce_axes'],
    ['reduce_sum_out_fp32'],
    keepdims=0
  )

  exp_node = helper.make_node('Exp', ['reduce_sum_out_fp32'], ['exp_out'])

  const2_val_fp16 = np.array(1.5, dtype=np.float16)
  const2_tensor = helper.make_tensor('const2_fp16', TensorProto.FLOAT16, [], const2_val_fp16.tobytes(), raw=True)
  add_node = helper.make_node('Add', ['exp_out', 'const2_fp16'], ['add_out'])

  const3_val_fp16 = np.array(50.0, dtype=np.float16)
  const3_tensor = helper.make_tensor('const3_fp16', TensorProto.FLOAT16, [], const3_val_fp16.tobytes(), raw=True)
  div_node = helper.make_node('Div', ['add_out', 'const3_fp16'], ['output'])

  output_shape = [input_shape[0], input_shape[2]]
  Y = helper.make_tensor_value_info('output', TensorProto.FLOAT16, output_shape)
  graph_def = helper.make_graph(
    [sin_node, mul_node, reduce_sum_node, exp_node, add_node, div_node],
    'test-model-half-reduce',
    [X],
    [Y],
    [const1_tensor, const2_tensor, reduce_axes_tensor, const3_tensor]
  )

  model_def = helper.make_model(graph_def, producer_name='tinygrad')
  checker.check_model(model_def)
  onnx.save(model_def, model_path)
  return model_path

class TestOnnxNumericalAccuracy(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    np.random.seed(1337)
    input_shape = (100, 20, 100)
    cls.model_path = create_onnx_model_half_precision_sum_reduce(input_shape)
    input_data_np = (np.random.rand(*input_shape)).astype(np.float16)
    cls.inputs = {'input': input_data_np}

  @classmethod
  def tearDownClass(cls):
    if os.path.exists(cls.model_path):
      os.remove(cls.model_path)

  def test_half_precision_sum_reduction_stable(self):
    # test fails with NUMERICAL_STABILITY=0
    # how do I disable cache so I can NUMERICAL_STABILITY=0 expected failure?
    with Context(NUMERICAL_STABILITY=1):
      validate(self.model_path, self.inputs, rtol=2e-3, atol=2e-3)

if __name__ == '__main__':
    unittest.main()