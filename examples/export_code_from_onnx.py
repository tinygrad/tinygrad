# Use tinygrad to export an ONNX model into code, compile it, and test it against the original ONNX model
# Does a simple .c file, a .rs file, a full Rust crate.  Could be adapted for safetenors or whatever

import os, sys
import numpy as np
import subprocess
from extra.onnx import get_run_onnx
from tinygrad import dtypes
from tinygrad.tensor import Tensor
from tinygrad.helpers import DEBUG
from tinygrad.device import Device
from extra.export_model import export_model, compile_net, jit_model
import onnx
import onnxruntime as ort
from onnx.helper import tensor_dtype_to_np_dtype
from extra.backends.rust import RUST_TYPE_MAP

def onnx_get_dimensions(onnx_tensor):
  tensor_data_type = onnx_tensor.type.tensor_type.elem_type
  data_type_str = tensor_dtype_to_np_dtype(tensor_data_type)
  shape = {"dims": [], "dtype": data_type_str}
  for dim in onnx_tensor.type.tensor_type.shape.dim:
    if dim.dim_value:
      shape["dims"].append(dim.dim_value)
  return shape

def onnx_get_shapes(onnx_model):
  inputs_shape = []
  for input_tensor in onnx_model.graph.input:
    input_shape = onnx_get_dimensions(input_tensor)
    inputs_shape.append(input_shape)
  outputs_shape = []
  for output_tensor in onnx_model.graph.output:
    output_shape = onnx_get_dimensions(output_tensor)
    outputs_shape.append(output_shape)
  if len(inputs_shape)!= 1:
    raise Exception("Only one input is supported")
  if len(outputs_shape)!= 1:
    raise Exception("Only one output is supported")
  if len(inputs_shape[0]["dims"]) != 2 or inputs_shape[0]["dims"][0] != 1:
    raise Exception("Input shape assumed to be [1, N]")
  if len(outputs_shape[0]["dims"]) != 2 or outputs_shape[0]["dims"][0] != 1:
    raise Exception("Output shape assumed to be [1, N]")
  return inputs_shape, outputs_shape

def onnx_test(onnx_model, np_input):
  s = onnx_model.SerializeToString()
  session = ort.InferenceSession(s)
  input_name = session.get_inputs()[0].name
  output = session.run(None, {input_name: np_input})
  return output[0][0]

def onnx_load_model(model_path):
  if model_path is None:
    print("Create dummy onnx model if none is provided")
    import torch
    class DummyModel(torch.nn.Module):
      def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(20, 4)
      def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    torch_model = DummyModel()
    dummy_input = torch.randn(1, 10)
    model_path = "/tmp/dummy_model.onnx"
    torch.onnx.export(torch_model, dummy_input, model_path)
  elif not os.path.exists(model_path):
    raise Exception(f"Model file {model_path} not found")
  return onnx.load(model_path)

# converts ONNX model to Tinygrad compatible model
class TinyOnnx:
  def __init__(self, onnx_model):
    self.xname = onnx_model.graph.input[0].name
    self.yname = onnx_model.graph.output[0].name
    self.run_onnx = get_run_onnx(onnx_model)

  def forward(self, x):
    return self.run_onnx({self.xname: x}, debug=False)[self.yname]

def tiny_model_run(tiny_model, the_input):
  run, _ = jit_model(tiny_model, the_input)
  output = run(the_input)
  return output[0].numpy()[0]

def clang_generate(c_code, bufs, bufs_to_save, inputs, outputs, encoded_weights):
  dtype_map = {dtypes.float: ("float",4)}
  input_name = list(inputs.keys())[0]
  output_name = list(outputs.keys())[0]
  input_type = dtype_map[bufs[input_name][1]]
  output_type = dtype_map[bufs[output_name][1]]
  input_len = int(inputs[input_name]//input_type[1])
  output_len = int(outputs[output_name]//output_type[1])
  wtype = input_type

  cprog = ["#include <string.h>", "#include <stdio.h>", "#include <stdlib.h>"]
  cprog += [c_code, ""]

  # weights
  if not encoded_weights:
    cprog += [f"void initialize({wtype[0]} *weights) {{"]
    weights = bytes()
    for name,cl in bufs_to_save.items():
      cprog.append(f"  memcpy({name}, weights + {len(weights)//wtype[1]}, {len(cl._buf)});")
      weights += bytes(cl._buf)
    cprog += ["}", ""]
    # write the weights to disk
    with open("/tmp/clang_weights", "wb") as f:
      f.write(weights)

  output_print = ["printf(\""]
  for _ in range(output_len-1):
    output_print.append("%f ")
  output_print.append("%f\\n\", ")
  for i in range(output_len-1):
    output_print.append(f"outputs[{i}], ")
  output_print.append(f"outputs[{output_len-1}]);")
  output_print = ''.join(output_print)

  # test program
  cprog += [f"int main(int argc, char *argv[]) {{"]

  if not encoded_weights:
    cprog += ["  // read in the weights from disk","  FILE *f = fopen(\"/tmp/clang_weights\", \"rb\");"]
    cprog += [f"  {wtype[0]} *weights = ({wtype[0]} *)malloc({len(weights)});",f"  fread(weights, 1, {len(weights)}, f);"]
    cprog += ["  fclose(f);", "","  // init the net","  initialize(weights);",""]
  
  cprog += ["  // test run",f"  {input_type[0]} input[{input_len}];"]
  cprog += [f"  {output_type[0]} outputs[{output_len}];",f"  for (int i = 0; i < {input_len}; i++) scanf(\"%f\", &input[i]);"]
  cprog += [f"  net(input, outputs);","",f"  {output_print}", "}"]

  # ready the program
  prg = '\n'.join(cprog)
  return prg

def rust_generate(rs_code, bufs, bufs_to_save, inputs, outputs, encoded_weights, crate, model_name="model", export_dir=""):
  input_name = list(inputs.keys())[0]
  output_name = list(outputs.keys())[0]
  input_type = RUST_TYPE_MAP[bufs[input_name][1]]
  output_type = RUST_TYPE_MAP[bufs[output_name][1]]
  input_len = int(inputs[input_name]//input_type[1])
  output_len = int(outputs[output_name]//output_type[1])
  wtype = input_type

  # test 
  rs_main = ["use std::fs::File;","use std::io::{self, Read};",""] if not encoded_weights else ["use std::io::{self};",""]
  if not crate: rs_main += [rs_code,""]
  rs_main += ["// Simple testing setup using stdin/stdout"]
  rs_main += ["fn main() -> io::Result<()> {"]
  rs_main += ["  // Initialize network","  let mut net = Net::new();",""]
  rs_main += [f"  // Create an input buffer of {input_len} {input_type[0]}s"]
  rs_main += [f"  let mut input = [0.0; {input_len}];",f"  let mut output = [0.0; {output_len}];","  let mut line = String::new();",""]
  if not encoded_weights:
    # write the weights to disk
    weights = bytes()
    for name,cl in bufs_to_save.items(): weights += bytes(cl._buf)
    with open("/tmp/rust_weights", "wb") as f:
      f.write(weights)
    rs_main += ["  // Read weights from a file","  let mut f = File::open(\"/tmp/rust_weights\")?;","  let mut weights_bytes = Vec::new();"]
    rs_main += ["  f.read_to_end(&mut weights_bytes)?;","",f"  // Convert bytes to {wtype[0]}"]
    rs_main += [f"  let mut weights: Vec<{wtype[0]}> = Vec::with_capacity(weights_bytes.len() / {wtype[1]});"]
    rs_main += ["  // Now map the weights_bytes into weights",f"  for i in 0..(weights_bytes.len()/{wtype[1]}) {{"]
    rs_main += [f"    weights.push({wtype[0]}::from_le_bytes([{','.join(['weights_bytes[i*4+'+str(i)+']' for i in range(wtype[1])])}]));","  }",""]
    rs_main += ["  // Initialize the network with weights","  net.initialize_weights(&weights);",""]
  rs_main += ["  // Get inputs","  for i in 0..input.len() {","    io::stdin().read_line(&mut line).unwrap();"]
  rs_main += ["    input[i] = line.trim().parse::<f32>().unwrap();","    line.clear();","  }",""]
  rs_main += ["  // Run the network","  net.run(&input, &mut output);","","  // Print the output"]
  rs_main += ["  let outputstr = output.iter().map(|item| item.to_string()).collect::<Vec<_>>().join(\" \");","  print!(\"{}\", outputstr);",""]
  rs_main += ["  Ok(())","}"]

  # export the code if not a crate, just as a string
  if not crate:
    prg = '\n'.join(rs_main)
    return prg

  # Isolate weights, if encoded, so we can put them in a separate file
  weights = []
  if encoded_weights:
    rs_code_new = [[],[]]
    for line in rs_code.split("\n"):
      if len(weights) == 0 and line != "// Encoded Weights":
        rs_code_new[0].append(line)
        continue
      if line == "": rs_code_new[1].append(line)
      if len(rs_code_new[1]) == 0: weights.append(f"pub {line}" if len(weights) != 0 else line)
      else: rs_code_new[1].append(line)
    rs_code = "\n".join(["use crate::weights::{*};",""] + rs_code_new[0] + rs_code_new[1])

  ## Make the main Rust crate
  crate_path = os.path.join(export_dir,model_name)
  os.makedirs(crate_path, exist_ok=True)
  crate_src_path = os.path.join(crate_path, "src")
  os.makedirs(crate_src_path, exist_ok=True)

  # Make main crate Cargo.toml file
  cargo_toml = ["[package]",f"name = \"{model_name}\"","version = \"0.1.0\"","authors = [\"<NAME> <<EMAIL>>\"]","edition = \"2021\"",""]
  with open(os.path.join(crate_path, "Cargo.toml"), "w") as f:
    f.write('\n'.join(cargo_toml))

  # Make the src/model.rs file
  with open(os.path.join(crate_src_path, "model.rs"), "w") as f:
      f.write("#![allow(unused_mut,unused_parens)]\n"+rs_code)

  if encoded_weights:
    # Make the src/weights.rs file
    with open(os.path.join(crate_src_path, "weights.rs"), "w") as f:
      f.write('\n'.join(weights))

  # Make the src/lib.rs file
  with open(os.path.join(crate_src_path, "lib.rs"), "w") as f:
    incs = ["mod weights;","mod model;","pub use model::Net;"] if encoded_weights else ["mod model;","pub use model::Net;"]
    f.write('\n'.join(incs))

  # Make the src/main.rs file
  with open(os.path.join(crate_src_path, "main.rs"), "w") as f:
    incs = ["mod weights;","mod model;","use model::Net;"] if encoded_weights else ["mod model;","use model::Net;"]
    f.write('\n'.join(incs+rs_main))

  ## Make a second crate to test the main crate as a library
  crate2_name = f"rlib_test_{model_name}"
  crate2_path = os.path.join(export_dir,crate2_name)
  os.makedirs(crate2_path, exist_ok=True)
  crate2_src_path = os.path.join(crate2_path, "src")
  os.makedirs(crate2_src_path, exist_ok=True)

  # Make main crate Cargo.toml file
  cargo_toml = ["[package]",f"name = \"{crate2_name}\"","version = \"0.1.0\"","authors = [\"<NAME> <<EMAIL>>\"]","edition = \"2021\"",""]
  cargo_toml += ["[dependencies]",f"{model_name} = {{ path = \"../{model_name}\" }}",""]
  with open(os.path.join(crate2_path, "Cargo.toml"), "w") as f:
    f.write('\n'.join(cargo_toml))

  # Make the src/main.rs file
  with open(os.path.join(crate2_src_path, "main.rs"), "w") as f:
    f.write('\n'.join(["use model::Net;"]+rs_main))

  return (crate_path, crate2_path)

def generate_src(tiny_model, device, the_input, encoded_weights, crate, model_name="", export_dir=""):
  model_code, inputs, outputs, _ = export_model(tiny_model, device, the_input, encoded_weights=encoded_weights)
  run, special_names = jit_model(tiny_model, the_input)
  _, _, bufs, bufs_to_save = compile_net(run, special_names)
  if device == "rust":
    return rust_generate(model_code, bufs, bufs_to_save, inputs, outputs, encoded_weights, crate, model_name=model_name, export_dir=export_dir)
  elif device == "clang":
    return clang_generate(model_code, bufs, bufs_to_save, inputs, outputs, encoded_weights)
  else:
    raise Exception(f"Unknown device {device}")

def rust_compile_src(prg):
  # Compile the source
  binary_path = "/tmp/compile_onnx_test"
  rustc_cmd = ['rustc']
  if int(os.environ.get('DEBUG',0)) < 2:
    rustc_cmd += ['-Aunused_parens','-Aunused_mut']
  rustc_cmd += ['-O', '-', '-o', binary_path]
  subprocess.check_output(rustc_cmd, input=prg.encode('utf-8'))
  return binary_path

def rust_compile_crate(crate_paths):
  basedir = os.path.abspath(os.path.curdir)
  binary_paths = []
  for crate_path in crate_paths:
    os.chdir(crate_path)
    subprocess.check_output(['cargo', 'build'])
    model_name = os.path.basename(crate_path)
    binary_paths.append(os.path.join(crate_path, "target", "debug", f"{model_name}"))
    os.chdir(basedir)
  return binary_paths

def clang_compile_src(prg):
  # add test weights
  binary_path = "/tmp/compile_onnx_test"
  subprocess.check_output(['clang', '-O2', '-lm', '-fPIC', '-x', 'c', '-', '-o', binary_path], input=prg.encode('utf-8'))
  return binary_path

def compile_src(src_code, device, crate):
  if device == "rust":
    if crate:
      return rust_compile_crate(src_code)
    else:
      return rust_compile_src(src_code)
  elif device == "clang":
    return clang_compile_src(src_code)
  else:
    raise Exception(f"Unknown device {device}")

def run_compiled_test(binary_path, the_input):
  c_input = '\n'.join(["%f" % x for x in the_input[0].numpy()])+"\n"
  c_output = [float(x) for x in subprocess.check_output([binary_path], input=c_input.encode('utf-8')).decode('utf-8').strip().split(" ")]
  return c_output

if __name__ == "__main__":
  import argparse
  SUPPORTED_LANGUAGES = ["rust", "clang"]

  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model", type=str, help="Path to onnx model file")
  parser.add_argument("--notest", action="store_true", help="Don't test the generated code")
  parser.add_argument("--save", action="store_true", help="Save the generated files")
  parser.add_argument("--crate", action="store_true", help="For Rust, generate a full crate")
  parser.add_argument("--export_dir", type=str, default="export", help="Where do we put the generated src code")
  parser.add_argument("-n", "--name", type=str, default="model", help="Name of the model")
  parser.add_argument("-w", "--weights", action="store_true", help="Encode weights in the generated src code")
  parser.add_argument("-l", "--language", type=str, default="clang", help=f"Device to compile for, one of {SUPPORTED_LANGUAGES}")
  args = parser.parse_args()

  # Set up the device/language settings
  if args.language not in SUPPORTED_LANGUAGES:
    raise Exception(f"Example only supports '{SUPPORTED_LANGUAGES}' not {args.language}")
  os.environ[args.language.upper()] = "1"
  Device.DEFAULT = args.language.upper()
  print(f"Compiling for {args.language}", file=sys.stderr)

  # load models
  onnx_model = onnx_load_model(args.model)
  tiny_model = TinyOnnx(onnx_model)

  # generate random input for onnx (np) and for tinygrad (Tensor)
  input_shapes, output_shapes = onnx_get_shapes(onnx_model)
  np.random.seed(123)
  np_input = np.random.randn(*input_shapes[0]["dims"]).astype(input_shapes[0]["dtype"])
  the_input = Tensor(np_input)

  if not args.notest:
    # run onnx model as the control
    onnx_output = onnx_test(onnx_model, np_input)
    print(f"onnx:     {onnx_output}", file=sys.stderr)

    # run tinygrad model
    tiny_output = tiny_model_run(tiny_model, the_input)
    print(f"tiny:     {tiny_output}", file=sys.stderr)
    np.testing.assert_allclose(onnx_output, tiny_output, atol=1e-5, rtol=1e-5)

  # compile the generated code
  src_code = generate_src(tiny_model, args.language, the_input, args.weights, args.crate, model_name=args.name, export_dir=args.export_dir)

  # save the generated code
  if args.save and not args.crate:
    with open(os.path.join(args.export_dir,f"model.{'rs' if args.language == 'rust' else 'c'}"), "w") as f:
      f.write(src_code)

  if args.notest: sys.exit() # we're done here

  # compile the generated code
  binary_path = compile_src(src_code, args.language, args.crate)
  # run the compiled code
  if not args.crate:
    compiled_output = run_compiled_test(binary_path, the_input)
    print(f"compiled: {compiled_output}   Testing {binary_path}", file=sys.stderr)
    np.testing.assert_allclose(onnx_output, compiled_output, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(tiny_output, compiled_output, atol=1e-5, rtol=1e-5)
  else:
    # run the crates
    crate_output = run_compiled_test(binary_path[0], the_input)
    print(f"crate:    {crate_output}   Testing {binary_path[0]}", file=sys.stderr)
    np.testing.assert_allclose(onnx_output, crate_output, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(tiny_output, crate_output, atol=1e-5, rtol=1e-5)
    rlib_output = run_compiled_test(binary_path[1], the_input)
    print(f"rlib:     {rlib_output}   Testing {binary_path[1]}", file=sys.stderr)
    np.testing.assert_allclose(onnx_output, rlib_output, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(tiny_output, rlib_output, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(crate_output, rlib_output, atol=1e-5, rtol=1e-5)

  print("Passed all tests")
