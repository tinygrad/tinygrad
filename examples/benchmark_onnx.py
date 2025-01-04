import sys, onnx, time, pathlib
from tinygrad import Tensor, TinyJit, Device, GlobalCounters, fetch, getenv
from extra.onnx import get_run_onnx
import onnxruntime as ort
import numpy as np

def get_input(inp:onnx.ValueInfoProto, config:dict):
  # TODO: not complete
  def _get_size(key:str) -> int:
    size = config.get("crop_size") or config.get("size") or 512
    if isinstance(size, int): return size
    elif isinstance(size, dict): return size.get(key)
    else: raise ValueError(f"{config} {key}")
  # get shape
  shape = []
  for x in inp.type.tensor_type.shape.dim:
    match (x.HasField("dim_value"), x.dim_param):
      case (True, _): shape.append(x.dim_value)
      case (False, "height"): shape.append(_get_size("height"))
      case (False, "width"): shape.append(_get_size("width"))
      case (False, "num_channels"): shape.append(config.get("in_channels", 3))
      case (False, "sequence_length"): shape.append(20)  # kinda random sequence length maybe use max_position_embeddings?
      case (False, "decoder_sequence_length"): shape.append(20) # dunno about these two lol
      case (False, "encoder_sequence_length"): shape.append(20) # dunno about these two lol
      case (False, _): shape.append(1)
  shape = tuple(shape)
  # get dtype
  dtype = onnx.helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type)
  # determine value
  match inp.name:
    case "input_ids":
      vocab_size = config.get("text_config", {}).get("vocab_size") or config.get("vocab_size", 50265)
      val = np.random.randint(0, vocab_size, shape)
    case "attention_mask": val = np.random.randint(0, 2, shape)
    case "token_type_ids": val = np.random.randint(0, config.get("type_vocab_size", 2), shape)
    case "image_tensor": val = np.random.randint(0, 256, shape)
    case _: val = (np.random.randn(*shape) * 8) if shape else np.array(0)
  return val.astype(dtype)

def benchmark(onnx_model_path:pathlib.Path, config:dict={}, test_vs_ort=False):
  print("running benchmark")
  onnx_model = onnx.load(onnx_model_path)
  Tensor.no_grad = True
  Tensor.training = False
  run_onnx = get_run_onnx(onnx_model)
  print("loaded into tinygrad onnx runner")

  # find preinitted tensors and ignore them
  initted_tensors = {inp.name for inp in onnx_model.graph.initializer}
  expected_inputs = [inp for inp in onnx_model.graph.input if inp.name not in initted_tensors]

  # get real inputs
  run_onnx_jit = TinyJit(lambda **kwargs: tuple(run_onnx({k:v.to(Device.DEFAULT) for k,v in kwargs.items()}).values()), prune=True)

  for i in range(3):
    new_inputs = {inp.name:Tensor(get_input(inp, config)).realize() for inp in expected_inputs}
    GlobalCounters.reset()
    print(f"run {i}")
    run_onnx_jit(**new_inputs)

  # run 20 times
  for _ in range(20):
    new_inputs = {inp.name:Tensor(get_input(inp, config)).realize() for inp in expected_inputs}
    GlobalCounters.reset()
    st = time.perf_counter()
    out = run_onnx_jit(**new_inputs)
    mt = time.perf_counter()
    out[0].realize(*out[1:])
    tiny_out = tuple(o.numpy() for o in out)
    et = time.perf_counter()
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {(et-st)*1e3:6.2f} ms")

  if test_vs_ort:
    sess = ort.InferenceSession(onnx_model_path)
    ort_out = sess.run([out.name for out in onnx_model.graph.output], {k:v.numpy() for k,v in new_inputs.items()})
    rtol, atol = 1e-3, 1e-3
    assert len(tiny_out) == len(ort_out)
    for tiny_v, ort_v in zip(tiny_out, ort_out):
      if tiny_v is None: assert tiny_v == ort_v
      else: np.testing.assert_allclose(tiny_v, ort_v, rtol=rtol, atol=atol)
    del sess
    print("ort test passed")

if __name__ == "__main__":
  onnx_file = fetch(sys.argv[1])
  print(f"loaded model {onnx_file}" )
  benchmark(onnx_file, int(getenv("ORT", "0")))