# An example to compile a small Tensorflow model to extremely portable C code

import os
import tensorflow as tf
import tf2onnx
import onnx
from examples.compile_efficientnet import compile_net
from extra.onnx import get_run_onnx
from tinygrad.tensor import Tensor

def get_uncompiled_model2(dataset_size=32, output_size=4):
  inputs = tf.keras.Input(shape=(dataset_size,), name="inputs")
  x = tf.keras.layers.Dense(16, activation="relu", name="dense_1")(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)
  outputs = tf.keras.layers.Dense(output_size, activation="sigmoid", name="predictions")(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model

def create_onnx_model():
  model = get_uncompiled_model2()
  input_signature = [tf.TensorSpec([1,32], tf.float32, name='x')]
  onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
  return onnx_model

def compile_onnx_model(onnx_model):
  run_onnx = get_run_onnx(onnx_model)

  from extra.jit import TinyJit
  @TinyJit
  def run(x): return run_onnx({"x": x}, debug=False)['predictions'].realize()

  the_input = Tensor.randn(1,32)
  the_output = run(the_input)
  the_output = run(the_input)

  special_names = {id(the_input.lazydata.realized.cl): "input", id(the_output.lazydata.realized.cl): "outputs"}
  cprog = compile_net(run, special_names)
  cprog[-1] = "return outputs;\n}"

  print('\n'.join(cprog).replace("void net()", "float *infer(float *input)").replace("float input[32];\n", ""))

if __name__ == "__main__":
  onnx_model = create_onnx_model()
  compile_onnx_model(onnx_model)

