# An example to compile a small Tensorflow model to extremely portable C code

import os, sys
os.environ["CLANG"] = '1'
os.environ["GPU"] = '1'

import numpy as np
import subprocess
import tensorflow as tf
import tf2onnx
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

def create_onnx_model(keras_model):
  input_signature = [tf.TensorSpec([1,32], tf.float32, name='x')]
  onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature, opset=13)
  return onnx_model

def compile_onnx_model(onnx_model):
  run_onnx = get_run_onnx(onnx_model)

  from tinygrad.jit import TinyJit
  @TinyJit
  def run(x): return run_onnx({"x": x}, debug=False)['predictions'].realize()

  the_input = Tensor.randn(1,32)
  the_output = run(the_input)
  the_output = run(the_input)

  special_names = {id(the_input.lazydata.realized.cl): "input", id(the_output.lazydata.realized.cl): "outputs"}
  cprog, statements, bufs, bufs_to_save = compile_net(run, special_names)
  cprog = ["#include <string.h>", "#include <stdio.h>", "#include <stdlib.h>"] + cprog

  # buffers (all except input)
  cprog += [f"float {x[0]}[{x[1]}];" for x in bufs.values() if x[0] != "input"]

  # weights
  cprog.append("void initialize(float *weights) {")
  weights = bytes()
  for name,cl in bufs_to_save.items():
    cprog.append(f"memcpy({name}, weights + {len(weights)//4}, {len(cl)});")
    weights += bytes(memoryview(cl)[0:len(cl)//4])
  cprog.append("}")

  # write the weights to disk
  with open("/tmp/tf_weights", "wb") as f:
    f.write(weights)

  # the net
  cprog += ["float *infer(float *input) {"] + statements + ["return outputs;", "}"]

  # test program
  cprog.append(f"""int main(int argc, char *argv[]) {{
    // read in the weights from disk
    FILE *f = fopen("/tmp/tf_weights", "rb");
    float *weights = (float *)malloc({len(weights)});
    fread(weights, 1, {len(weights)}, f);
    fclose(f);

    // init the net
    initialize(weights);

    // test run
    float input[32];
    for (int i = 0; i < 32; i++) scanf("%f", &input[i]);
    float *outputs = infer(input);
    printf("%f %f %f %f\\n", outputs[0], outputs[1], outputs[2], outputs[3]);
  }}""")

  # ready the program
  prg = '\n'.join(cprog)
  print(prg)

  # add test weights
  subprocess.check_output(['clang', '-O2', '-lm', '-fPIC', '-x', 'c', '-', '-o', "/tmp/tf_test"], input=prg.encode('utf-8'))

  tinygrad_output = [x for x in the_output.numpy()[0]]
  print("tinygrad:", tinygrad_output, file=sys.stderr)

  c_input = ' '.join(["%f" % x for x in the_input[0].numpy()])+"\n"
  c_output = [float(x) for x in subprocess.check_output(["/tmp/tf_test"], input=c_input.encode('utf-8')).decode('utf-8').strip().split(" ")]
  print("compiled:", c_output, file=sys.stderr)

  np.testing.assert_allclose(tinygrad_output, c_output, atol=1e-5, rtol=1e-5)
  return the_input.numpy(), c_output

if __name__ == "__main__":
  keras_model = get_uncompiled_model2()
  onnx_model = create_onnx_model(keras_model)
  test_input, test_output = compile_onnx_model(onnx_model)
  tf_output = keras_model(test_input).numpy()[0]
  print("keras:   ", tf_output, file=sys.stderr)
  np.testing.assert_allclose(tf_output, test_output, atol=1e-5, rtol=1e-5)

