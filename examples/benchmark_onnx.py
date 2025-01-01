from typing import Literal
import sys, onnx, time, pathlib, urllib.error, json
import numpy as np
from tinygrad import Tensor, TinyJit, Device, GlobalCounters, fetch, getenv
from extra.onnx import get_run_onnx
import onnxruntime as ort

def benchmark(onnx_file:pathlib.Path):
  print(f"running benchmark")
  onnx_model = onnx.load(onnx_file)
  Tensor.no_grad = True
  Tensor.training = False
  run_onnx = get_run_onnx(onnx_model)
  print("loaded model")

  # find preinitted tensors and ignore them
  initted_tensors = {inp.name for inp in onnx_model.graph.initializer}
  expected_inputs = [inp for inp in onnx_model.graph.input if inp.name not in initted_tensors]

  # get real inputs
  model_config_path = onnx_file.parent.joinpath("config.json")
  preprocessor_config_path = onnx_file.parent.joinpath("preprocessor_config.json")
  model_config = json.load(model_config_path.open()) if model_config_path.exists() else {}
  preprocessor_config = json.load(preprocessor_config_path.open()) if preprocessor_config_path.exists() else {}
  def get_input(inp:onnx.ValueInfoProto) -> tuple:
    # TODO: not complete
    def _get_size(key:str) -> int:
      size = preprocessor_config.get("crop_size") or preprocessor_config.get("size") or 512
      if isinstance(size, int): return size
      elif isinstance(size, dict): return size.get(key)
      else: raise ValueError(f"{preprocessor_config} {key}")
    # get shape
    shape = []
    for x in inp.type.tensor_type.shape.dim:
      match (x.HasField("dim_value"), x.dim_param):
        case (True, _): shape.append(x.dim_value)
        case (False, "height"): shape.append(_get_size("height"))
        case (False, "width"): shape.append(_get_size("width"))
        case (False, "num_channels"): shape.append(model_config.get("in_channels", 3))
        case (False, "sequence_length"): shape.append(20)  # kinda random sequence length maybe use max_position_embeddings?
        case (False, "decoder_sequence_length"): shape.append(20) # dunno about these two lol
        case (False, "encoder_sequence_length"): shape.append(20) # dunno about these two lol
        case (False, _): shape.append(1)
    shape = tuple(shape)
    # get dtype
    dtype = onnx.helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type)
    # determine value
    # TODO: use numpy here for now to check if float16?
    match inp.name:
      case "input_ids":
        vocab_size = model_config.get("text_config", {}).get("vocab_size") or model_config.get("vocab_size", 50265)
        val = np.random.randint(0, vocab_size, shape)
      case "attention_mask": val = np.random.randint(0, 2, shape)
      case "token_type_ids": val = np.random.randint(0, model_config.get("type_vocab_size", 2), shape)
      case _: val = np.random.randn(*shape) if shape else np.array(0)
    return val.astype(dtype)

  run_onnx_jit = TinyJit(lambda **kwargs: tuple(run_onnx({k:v.to(Device.DEFAULT) for k,v in kwargs.items()}).values()), prune=True)

  for i in range(3):
    new_inputs = {inp.name:Tensor(get_input(inp)).realize() for inp in expected_inputs}
    GlobalCounters.reset()
    print(f"run {i}")
    run_onnx_jit(**new_inputs)

  # run 20 times
  for _ in range(20):
    new_inputs = {inp.name:Tensor(get_input(inp)).realize() for inp in expected_inputs}
    GlobalCounters.reset()
    st = time.perf_counter()
    out = run_onnx_jit(**new_inputs)
    mt = time.perf_counter()
    out[0].realize(*out[1:])
    tiny_out = tuple(o.numpy() for o in out)
    et = time.perf_counter()
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {(et-st)*1e3:6.2f} ms")

  if test_with_ort:
    # TODO: I'm fudging up some memory management thing (JIT?) I think. I'm not sure what's going on. so I rerun it here again
    new_inputs = {inp.name:get_input(inp) for inp in expected_inputs}
    run_onnx = get_run_onnx(onnx_model)
    tiny_out = tuple(run_onnx(new_inputs).values())

    sess = ort.InferenceSession(onnx_file)
    ort_out = sess.run([o.name for o in onnx_model.graph.output], new_inputs)
    # TODO: how do we determine rtol atol?
    rtol, atol = 1e-3, 1e-3
    assert len(tiny_out) == len(ort_out)
    for tiny_v, ort_v in zip(tiny_out, ort_out):
      if tiny_v is None: assert tiny_v == ort_v
      else: np.testing.assert_allclose(tiny_v.numpy(), ort_v, rtol=rtol, atol=atol)
    print("ort test passed")

def download(potential_file_paths:list[str], model_id:str):
  """ downloads the model, external data, config, and preprocessing config from huggingface """
  base_path = f"https://huggingface.co/{model_id}/resolve/main"
  for file_path in potential_file_paths:
    url = f"{base_path}/{file_path}"
    model_name = file_path.split('/')[-1]

    # download onnx model
    try:
      model_path = fetch(url, model_name, model_id)
      print(f"Downloaded model at {model_path.as_posix()}")
    # early continue to the next file model isn't found
    except urllib.error.HTTPError as e:
      if e.code == 404: continue
      raise
    # raise error if unexpected error occurs
    except Exception: raise

    # download onnx external data in the same directory
    try:
      file_path_no_extension = url.rsplit('.', 1)[0]
      external_data_url = f"{file_path_no_extension}.onnx_data"
      external_data_name = external_data_url.split('/')[-1]
      external_data_path = fetch(external_data_url, external_data_name, model_id)
      print(f"Downloaded external data at {external_data_path.as_posix()}")
    except urllib.error.HTTPError as e:
      if e.code != 404: raise
    except Exception: raise

    # download configs
    for config_path in (base_path, base_path + "/onnx"):
      try:
        preprocessor_config = fetch(f"{config_path}/preprocessor_config.json", "preprocessor_config.json", model_id)
        print(f"Downloaded preprocessor config at {preprocessor_config.as_posix()}")
        break
      except urllib.error.HTTPError as e:
        if e.code != 404: raise
      except Exception: raise
    for config_path in (base_path, base_path + "/onnx"):
      try:
        model_config = fetch(f"{config_path}/config.json", "config.json", model_id)
        print(f"Downloaded model config at {model_config.as_posix()}")
        break
      except urllib.error.HTTPError as e:
        if e.code != 404: raise
      except Exception: raise

    # yield the model path
    yield model_path

  raise Exception(f"failed to download model from https://huggingface.co/{model_id}")

POTENTIAL_MODEL_PATHS = [
  "onnx/model.onnx",
  "model.onnx",
  "onnx/decoder_model.onnx",
  "onnx/decoder_model_merged.onnx",
  "punct_cap_seg_en.onnx", # for "1-800-BAD-CODE/punctuation_fullstop_truecase_english"
]

def benchmark_from_huggingface(sort:Literal["downloads", "download_all_time", "trending"]="downloads", limit:int=100):
  from huggingface_hub import list_models
  # TODO: should we just download all onnx models and files? Then optionally run them? Less hacks this way
  # NOTE: we only download 1 model per project. Can just download all of them.
  for i, model in enumerate(list_models(filter="onnx", sort=sort, limit=limit)):
    # TODO: uses a pipeline of different models with configs scattered everywhere
    if model.id in {"stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/sdxl-turbo"}: continue
    # TODO: `HuggingFaceTB/SmolLM2-360M-Instruct` need `GroupQueryAttention`
    # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftgroupqueryattention
    if model.id in {"HuggingFaceTB/SmolLM2-360M-Instruct"}: continue
    # TODO: mod is still not quite right, or may be something else
    # File "/Users/zibo/fun/tiny/tinygrad/tinygrad/ops.py", line 661, in <lambda>
    # Ops.MOD: lambda x,y: abs(int(x))%abs(int(y))*(1,-1)[x<0], Ops.IDIV: lambda x,y: abs(x)//abs(y)*(1,-1)[x*y<0] if y != 0 else 0,
    # ZeroDivisionError: integer division or modulo by zero
    if model.id in {"briaai/RMBG-2.0"}: continue

    print(f"{i}: {model.id} ({model.downloads} {sort}) ")
    print(f"link: https://huggingface.co/{model.id}")
    model_path = next(download(POTENTIAL_MODEL_PATHS, model.id))
    benchmark(model_path)

def benchmark_model_id(model_id:str):
  model_path = next(download(POTENTIAL_MODEL_PATHS, model_id))
  benchmark(model_path)

if __name__ == "__main__":
  # sort` options:
  # "downloads": recent 30-day total number of downloads
  # "download_all_time": all-time total number of downloads
  # "trending": some trending metric huggingface uses idk
  sort = getenv("SORT", "downloads")
  limit = int(getenv("LIMIT", "100"))
  test_with_ort = int(getenv("ORT", "0"))
  single_model = getenv("MODEL", "") # bench a single model using the model id e.g. "HuggingFaceTB/SmolLM2-360M-Instruct"

  if len(sys.argv) > 1:
    print(f"** Running benchmark for {sys.argv[1]} **")
    onnx_file = fetch(sys.argv[1])
    benchmark(onnx_file)
  elif single_model != "":
    print(f"** Running benchmark for {single_model} on huggingface **")
    benchmark_model_id(single_model)
  else:
    print(f"** Running benchmarks for top {limit} models ranked by {sort} on huggingface **")
    benchmark_from_huggingface(sort, limit)