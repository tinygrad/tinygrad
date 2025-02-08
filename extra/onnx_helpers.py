from tinygrad import Tensor
from tinygrad.tensor import _to_np_dtype
from extra.onnx import OnnxRunner, OnnxValue
import onnx, os
import numpy as np
import onnxruntime as ort

def get_example_inputs(graph_inputs:dict[str, OnnxValue], config):
  def _get_shape(shape: tuple[str|int]):
    ret = []
    for dim in shape:
      match dim:
        case int(): ret.append(dim)
        case "width" | "height": ret.append(224)
        case "sequence_length": ret.append(20)
        case "decoder_sequence_length" | "encoder_sequence_length": ret.append(20)
        case "past_decoder_sequence_length" | "encoder_sequence_length_out": ret.append(20)
        case "batch_size": ret.append(1)
        case "num_channels": ret.append(config.get("in_channels", 3))
        case _: ret.append(1)
    return ret
  def _get_value(name, shape, dtype):
    match name:
      case "input_ids":
        vocab_size = config.get("text_config", {}).get("vocab_size") or config.get("vocab_size", 50265)
        val = np.random.randint(0, vocab_size, shape)
      case "attention_mask": val = np.random.randint(0, 2, shape)
      case "token_type_ids": val = np.random.randint(0, config.get("type_vocab_size", 2), shape)
      case "image_tensor": val = np.random.randint(0, 256, shape)
      case _: val = np.random.uniform(size=shape) * 8
    return val.astype(_to_np_dtype(dtype))

  ret: dict[str, Tensor] = {}
  for name, spec in graph_inputs.items():
    assert not spec.is_optional and not spec.is_sequence, "only allow tensor input for now"
    shape = _get_shape(spec.shape)
    value = Tensor(_get_value(name, shape, spec.dtype)).realize()
    ret.update({name:value})
  return ret

def slice_model(onnx_file, limit:int):
  model = onnx.load(onnx_file, load_external_data=False)
  nodes_up_to_limit = list(model.graph.node)[:limit+1]
  new_output_values = [onnx.helper.make_empty_tensor_value_info(output_name) for output_name in nodes_up_to_limit[-1].output]
  model.graph.ClearField("node")
  model.graph.node.extend(nodes_up_to_limit)
  model.graph.ClearField("output")
  model.graph.output.extend(new_output_values)
  base, ext = os.path.splitext(onnx_file)
  new_onnx_file = f"{base}_limit_{limit}{ext}"
  onnx.save_model(model, new_onnx_file)
  return new_onnx_file

def validate(onnx_file, inputs:dict|None=None, limit:int=-1, rtol=1e-5, atol=1e-5):
  if limit != -1: onnx_file = slice_model(onnx_file, limit)
  run_onnx = OnnxRunner(onnx.load(onnx_file))
  if inputs is None: inputs = get_example_inputs(run_onnx.graph_inputs)
  tinygrad_out = run_onnx(inputs)

  ort_options = ort.SessionOptions()
  ort_options.log_severity_level = 3
  ort_sess = ort.InferenceSession(onnx_file, ort_options, ["CPUExecutionProvider"])
  np_inputs = {k:v.numpy() if isinstance(v, Tensor) else v for k,v in inputs.items()}
  out_names = list(run_onnx.graph_outputs)
  out_values = ort_sess.run(out_names, np_inputs)
  ort_out = dict(zip(out_names, out_values))

  assert len(tinygrad_out) == len(ort_out) and tinygrad_out.keys() == ort_out.keys()
  for k in tinygrad_out.keys():
    tiny_v, onnx_v = tinygrad_out[k], ort_out[k]
    if tiny_v is None: assert tiny_v == onnx_v
    else: np.testing.assert_allclose(tiny_v.numpy(), onnx_v, rtol=rtol, atol=atol, err_msg=f"For tensor '{k}' in {tinygrad_out.keys()}")

  if limit != -1: os.remove(onnx_file)
