import urllib.error, json
from typing import Literal
from collections import Counter

from extra.onnx import get_run_onnx
from tinygrad.helpers import fetch, getenv
from test.external.external_model_benchmark import assert_allclose
from huggingface_hub import list_models
import numpy as np
import onnx
from onnx.external_data_helper import uses_external_data, load_external_data_for_model
from onnx.helper import tensor_dtype_to_np_dtype
import onnxruntime as ort
ort_options = ort.SessionOptions()
ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_options.log_severity_level = 3  # no warnings
ort_options.intra_op_num_threads = 1

total_opset = Counter()

# non-quantized model path ranked by try priority
POTENTIAL_MODEL_PATHS = [
  "onnx/model.onnx",
  "model.onnx",
  "onnx/decoder_model.onnx",
  "onnx/decoder_model_merged.onnx",
  "unet/model.onnx", # for "stabilityai/sdxl-turbo" and "stabilityai/stable-diffusion-xl-base-1.0"
  "punct_cap_seg_en.onnx", # for "1-800-BAD-CODE/punctuation_fullstop_truecase_english"
]

def download_onnx_model(model_id:str, root_url:str):
  for model_path in POTENTIAL_MODEL_PATHS:
    url = root_url + '/' + model_path
    try: return fetch(url, model_path.split('/')[-1], model_id.split("/")[-1]), url
    except urllib.error.HTTPError: pass
  return None, None

def download_metadata(urls:list[str], file_name:str):
  for url in urls:
    try: return fetch(url + '/' + file_name)
    except urllib.error.HTTPError: pass
  return None

def check_require_external_data(model_proto:onnx.ModelProto) -> bool:
  for initializer in model_proto.graph.initializer:
    if uses_external_data(initializer): return True
  return False

def _get_size(preprocessor_config:dict, key:str) -> int:
  size = preprocessor_config.get("crop_size") or preprocessor_config.get("size") or 512
  if isinstance(size, int): return size
  elif isinstance(size, dict): return size.get(key)
  else: raise ValueError(f"{preprocessor_config} {key}")
def get_input(inp:onnx.ValueInfoProto, model_config:dict, preprocessor_config:dict) -> tuple:
  # get shape
  shape = []
  for x in inp.type.tensor_type.shape.dim:
    match (x.HasField("dim_value"), x.dim_param):
      case (True, _): shape.append(x.dim_value)
      case (False, "height"): shape.append(_get_size(preprocessor_config, "height"))
      case (False, "width"): shape.append(_get_size(preprocessor_config, "width"))
      case (False, "num_channels"): shape.append(model_config.get("in_channels", 3))
      case (False, "sequence_length"): shape.append(20)  # kinda random sequence length maybe use max_position_embeddings?
      case (False, _): shape.append(1)
  shape = tuple(shape)

  # get dtype
  dtype = tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type)

  # determine value
  match inp.name:
    case "input_ids":
      vocab_size = model_config.get("text_config", {}).get("vocab_size") or model_config.get("vocab_size", 50265)
      val = np.random.randint(0, vocab_size, shape)
    case "attention_mask": val = np.random.randint(0, 2, shape)
    case "token_type_ids": val = np.random.randint(0, model_config.get("type_vocab_size", 2), shape)
    case _: val = np.random.randn(*shape) if shape else np.array(0)
  return val.astype(dtype)

def benchmark_model(model_id: str):
  root_url = f"https://huggingface.co/{model_id}/resolve/main"
  model_fp, model_url = download_onnx_model(model_id, root_url)
  base_url, file_name = model_url.rsplit('/', 1)

  if model_fp is None: raise Exception(f"failed to download from https://huggingface.co/{model_id}")
  onnx_model = onnx.load(model_fp, load_external_data=False)
  if check_require_external_data(onnx_model):
    file_name_root = file_name.split(".")[0]
    data_url = base_url + '/' + f'{file_name_root}.onnx_data'
    data_fp = fetch(data_url, f"{file_name_root}.onnx_data", model_id.split("/")[1])
    load_external_data_for_model(onnx_model, data_fp.parent.as_posix())

  # prepare data
  output_names = [out.name for out in onnx_model.graph.output]
  excluded = {inp.name for inp in onnx_model.graph.initializer}

  preprocessor_config = download_metadata([base_url, root_url], "preprocessor_config.json")
  try: preprocessor_config = json.load(preprocessor_config.open())
  except: preprocessor_config = {}
  model_config = download_metadata([base_url, root_url], "config.json")
  try: model_config = json.load(model_config.open())
  except: model_config = {}

  np_inputs = {inp.name:get_input(inp, model_config, preprocessor_config) for inp in onnx_model.graph.input if inp.name not in excluded}

  # run tinygrad
  tinygrad_runner = get_run_onnx(onnx_model)
  tinygrad_out = tinygrad_runner(np_inputs)
  tinygrad_out = {k:v.realize() for k,v in tinygrad_out.items()}

  # run ort
  # TODO: ort narrowing error
  # https://github.com/microsoft/onnxruntime/discussions/22736#discussioncomment-11411120
  # skip running ort and verification for now
  if model_id in {"FacebookAI/xlm-roberta-large", "BAAI/bge-m3", "intfloat/multilingual-e5-large", "BAAI/bge-reranker-large",
                  "FacebookAI/xlm-roberta-large-finetuned-conll03-english", "MoritzLaurer/bge-m3-zeroshot-v2.0", "jinaai/jina-embeddings-v3"
                  "oliverguhr/fullstop-punctuation-multilang-large", "oliverguhr/fullstop-punctuation-multilang-large",
                  "intfloat/multilingual-e5-large-instruct", "bigscience/bloom-560m", "jinaai/jina-embeddings-v3",
                  "openai-community/gpt2-large"}: return
  ort_sess = ort.InferenceSession(onnx_model.SerializeToString(deterministic=True), ort_options, ["CPUExecutionProvider"])
  ort_out = ort_sess.run(output_names, np_inputs)
  ort_out = dict(zip(output_names, ort_out))
  del ort_sess, onnx_model

  # validate
  assert_allclose(tinygrad_out, ort_out, rtol=2e-3, atol=2e-3)

def benchmark_top_models(sort:Literal["downloads", "download_all_time"]="downloads", limit=100):
  """
  NOTE: for sort:
    - 'downloads' (int): 30-day number of downloads
    - 'downloads_all_time' (int): All-time number of downloads
  """
  for i, model in enumerate(list_models(filter="onnx", sort=sort, limit=limit)):
    # TODO: uses a pipeline of different models with configs scattered everywhere
    if model.id in {"stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/sdxl-turbo"}: continue
    # TODO: `HuggingFaceTB/SmolLM2-360M-Instruct` need `GroupQueryAttention`
    # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftgroupqueryattention
    if model.id in {"HuggingFaceTB/SmolLM2-360M-Instruct"}: continue
    # TODO: need MOD!!!!!!!
    if model.id in {"briaai/RMBG-2.0"}: continue
    # TODO: no clue. Needs to be fixed.
    # File "/Users/zibo/fun/tiny/tinygrad/tinygrad/dtype.py", line 96, in as_const
    # return int(val) if dtypes.is_int(dtype) else float(val) if dtypes.is_float(dtype) else bool(val)
    # OverflowError: cannot convert float infinity to integer
    # NOTE: increasing the sequence_length variable dim from 1 to 20 or realizing before cast will fixed this.
    # still no idea why it failed in the first place
    # if model.id in {"sentence-transformers/all-mpnet-base-v2", "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    #                 "sentence-transformers/paraphrase-mpnet-base-v2", "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    #                 "sentence-transformers/nli-mpnet-base-v2"}: continue

    print(f"{i:<5} {model.id:<50} {model.downloads:<10}")
    benchmark_model(model.id)

if __name__ == "__main__":
  if input_model_id := getenv("MODEL", ""):
    benchmark_model(input_model_id)
  else:
    benchmark_top_models(limit=100)
