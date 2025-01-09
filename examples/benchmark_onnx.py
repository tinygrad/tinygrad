import sys, onnx, time, pathlib
from tinygrad import Tensor, TinyJit, Device, GlobalCounters, fetch, getenv
from extra.onnx import get_run_onnx, dtype_parse
import onnxruntime as ort
import numpy as np

def get_input(inp:onnx.ValueInfoProto, config:dict) -> Tensor:
  # TODO: not complete
  def _get_size(key:str) -> int:
    size = config.get("crop_size") or config.get("size") or 224
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
      case (False, "sequence_length"): shape.append(20)  # maybe use max_position_embeddings?
      case (False, "decoder_sequence_length"): shape.append(20)
      case (False, "encoder_sequence_length"): shape.append(20)
      case (False, _): shape.append(1)
  # get dtype
  dtype = dtype_parse(inp.type.tensor_type.elem_type)
  # determine value
  match inp.name:
    case "input_ids":
      vocab_size = config.get("text_config", {}).get("vocab_size") or config.get("vocab_size", 50265)
      val = Tensor.randint(*shape, low=0, high=vocab_size, dtype=dtype)
    case "attention_mask": val = Tensor.randint(*shape, low=0, high=2, dtype=dtype)
    case "token_type_ids": val = Tensor.randint(*shape, low=0, high=config.get("type_vocab_size", 2), dtype=dtype)
    case "image_tensor": val = Tensor.randint(*shape, low=0, high=256, dtype=dtype)
    case _: val = Tensor.randn(*shape, dtype=dtype).mul(8) if shape else Tensor(None, dtype=dtype)
  return val.realize()

def benchmark(onnx_model_path:pathlib.Path, config:dict={}, test_vs_ort=False):
  print("running benchmark")
  onnx_model = onnx.load(onnx_model_path)
  Tensor.no_grad = True
  Tensor.training = False
  run_onnx = get_run_onnx(onnx_model)
  print("loaded model")

  # find preinitted tensors and ignore them
  initted_tensors = {inp.name for inp in onnx_model.graph.initializer}
  expected_inputs = [inp for inp in onnx_model.graph.input if inp.name not in initted_tensors]

  # get real inputs
  run_onnx_jit = TinyJit(lambda **kwargs: tuple(run_onnx({k:v.to(Device.DEFAULT) for k,v in kwargs.items()}).values()), prune=True)

  for i in range(3):
    new_inputs = {inp.name:get_input(inp, config) for inp in expected_inputs}
    GlobalCounters.reset()
    print(f"run {i}")
    run_onnx_jit(**new_inputs)

  # run 20 times
  for _ in range(20):
    new_inputs = {inp.name:get_input(inp, config) for inp in expected_inputs}
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
  benchmark(onnx_file, test_vs_ort=int(getenv("ORT", "0")))