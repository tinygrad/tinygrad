from tinygrad import Tensor
from tinygrad.tensor import _to_np_dtype
from tinygrad.nn.onnx import OnnxRunner, OnnxValue
import numpy as np
import onnxruntime as ort
ort_options = ort.SessionOptions()
ort_options.log_severity_level = 3

def get_example_inputs(graph_inputs:dict[str, OnnxValue], config={}):
  def _get_shape(onnx_shape: tuple[str|int]):
    shape = []
    for onnx_dim in onnx_shape:
      match onnx_dim:
        case int(): shape.append(onnx_dim)
        case "width" | "height":
          size = config.get("size", {})
          shape.append(size) if isinstance(size, int) else shape.append(size.get(onnx_dim, 224))
        case "sequence" | "sequence_length" | "decoder_sequence_length": shape.append(64)
        case "encoder_sequence_length": shape.append(config.get("nb_max_frames", 64))
        case "past_decoder_sequence_length" | "encoder_sequence_length_out": shape.append(64)
        case "encoder_sequence_length / 2": shape.append(32)
        case "batch_size": shape.append(1)
        case "num_channels": shape.append(config.get("in_channels", 3))
        case "num_channels_latent": shape.append(config.get("latent_channels", 4))
        case "height_latent" | "width_latent": shape.append(config.get("sample_size", 1024) // 8)
        case "feature_size": shape.append(config.get("num_mel_bins", 128))
        case _: shape.append(1)
    return shape
  def _get_value(name, shape, dtype):
    match name:
      case "input_ids":
        vocab_size = config.get("text_config", {}).get("vocab_size") or config.get("vocab_size", 32)
        val = np.random.randint(0, vocab_size-1, shape)
      case "attention_mask": val = np.random.randint(0, 2, size=shape)
      case "token_type_ids": val = np.random.randint(0, config.get("type_vocab_size", 2), shape)
      case "image_tensor": val = np.random.randint(0, 256, shape)
      case "task_id": return Tensor(0, dtype=dtype)
      case _: val = np.random.uniform(size=shape) * 8
    return Tensor(val.astype(_to_np_dtype(dtype))).realize()

  ret: dict[str, Tensor] = {}
  for name, spec in graph_inputs.items():
    assert not spec.is_optional and not spec.is_sequence, "only allow tensor input for now"
    shape = _get_shape(spec.shape)
    value = _get_value(name, shape, spec.dtype)
    ret.update({name:value})
  return ret

def _get_tinygrad_and_ort_np_outputs(onnx_file, inputs):
  run_onnx = OnnxRunner(onnx_file)

  ort_sess = ort.InferenceSession(onnx_file, ort_options, ["CPUExecutionProvider"])
  np_inputs = {k:v.numpy() if isinstance(v, Tensor) else v for k,v in inputs.items()}
  out_names = list(run_onnx.graph_outputs)
  out_values = ort_sess.run(out_names, np_inputs)
  ort_out = dict(zip(out_names, out_values))

  tinygrad_out = run_onnx(inputs)
  Tensor.realize(*tinygrad_out.values())
  tinygrad_out = {k:v.numpy() for k,v in tinygrad_out.items()}
  return tinygrad_out, ort_out

def validate(onnx_file, inputs, rtol=1e-5, atol=1e-5):
  tinygrad_out, ort_out = _get_tinygrad_and_ort_np_outputs(onnx_file, inputs)

  assert tinygrad_out.keys() == ort_out.keys()
  for k in tinygrad_out.keys():
    tiny_v, onnx_v = tinygrad_out[k], ort_out[k]
    if tiny_v is None: assert onnx_v is None, f"{k}: {tiny_v=}, {onnx_v=}"
    else: np.testing.assert_allclose(tiny_v, onnx_v, rtol=rtol, atol=atol, err_msg=f"For tensor '{k}' in {tinygrad_out.keys()}")

def validate_all_intermediates(onnx_file, inputs, rtol=1e-5, atol=1e-5):
  report = produce_node_output_report(onnx_file, inputs)
  for i, node in enumerate(report):
    node_name = node["node"]
    outputs = node["outputs"]
    print(f"Validating node_{i}: {node_name}")
    for output in outputs:
      output_name = output["name"]
      tinygrad_out = output["tinygrad"]
      ort_out = output["onnxruntime"]
      np.testing.assert_allclose(tinygrad_out, ort_out, rtol=rtol, atol=atol, err_msg=f"For tensor '{output_name}' in {node_name}")
      abs_diff = np.abs(tinygrad_out - ort_out)
      ort_out_abs = np.abs(ort_out)
      relative_diff = np.divide(abs_diff, ort_out_abs, out=np.zeros_like(abs_diff), where=(ort_out_abs != 0))
      print(f"{output_name}: (max_abs_diff={np.max(abs_diff):.2e}, max_rel_diff={np.max(relative_diff):.2e})")

def produce_node_output_report(onnx_file, inputs):
  """
  Build a report of all ONNX node outputs from tinygrad and onnxruntime

  Returns:
  ```
  [
    {
      "node": str,
      "outputs": [
        { "name": str, "tinygrad": np.ndarray, "onnxruntime": np.ndarray },
        ...
      ]
    },
    ...
  ]
  ```

  Note:
  - requires onnx_graphsurgeon: pip install onnx-graphsurgeon
  """
  import onnx_graphsurgeon as gs
  import onnx
  import tempfile

  # rewrite the model to output all the node outputs
  # infer shapes here fills the shapes and dtypes of intermediate values which graphsurgeon requires when assigning them as outputs
  inferred_model = onnx.shape_inference.infer_shapes(onnx.load(onnx_file))
  model = gs.import_onnx(inferred_model)
  model_nodes = model.nodes
  node_outputs = [n.outputs for n in model.nodes]
  model.outputs = [each_output for outputs in node_outputs for each_output in outputs]
  rewritten_model = gs.export_onnx(model)
  # TODO: remove this once ORT supports 1.18.0
  if getattr(rewritten_model, "ir_version", 0) > 10:
    rewritten_model.ir_version = 10
  with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as f:
    onnx.save(rewritten_model, f.name)
    rewritten_model = f.name

  tinygrad_out, ort_out = _get_tinygrad_and_ort_np_outputs(rewritten_model, inputs)

  report = []
  for node in model_nodes:
    outputs = []
    for each_output in node.outputs:
      name = each_output.name
      outputs.append({ "name": name, "tinygrad": tinygrad_out[name], "onnxruntime": ort_out[name] })
    report.append({"node": node.name, "outputs": outputs})

  return report


if __name__ == "__main__":
  from tinygrad.helpers import fetch
  fn = fetch("https://github.com/commaai/openpilot/raw/v0.9.4/selfdrive/modeld/models/supercombo.onnx")
  validate(fn, get_example_inputs(OnnxRunner(fn).graph_inputs), rtol=7e-3, atol=9e-2)
  # pprint(report)
