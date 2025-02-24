import onnx, json, tempfile
from pathlib import Path
from huggingface_hub import list_models, snapshot_download
from tinygrad.helpers import _ensure_downloads_dir, getenv
from extra.onnx import OnnxRunner
from extra.onnx_helpers import validate, get_example_inputs

HUGGINGFACE_URL = "https://huggingface.co"
SKIPPED_FILES = [
  "avx2", "arm64", "avx512", "avx512_vnni", # hardware specific and DynamicDequantizeLinear gives numerically inaccurate values
  "q4", "q4f16", "bnb4", # other unimplemented quantization
  "model_O4", # requires non cpu ort runner and MemcpyFromHost
  "merged", # TODO implement attribute with graph type
  "fp16", "int8", "uint8", "quantized", # numerical accuracy issues
]
SKIPPED_REPO_PATHS = [
  "mangoapps/fb_zeroshot_mnli_onnx", # implement NonZero op
  "minishlab/potion-base-8M", # implement attribute with graph type
  "HuggingFaceTB/SmolLM2-360M-Instruct", # TODO implement GroupQueryAttention
  "HuggingFaceTB/SmolLM2-1.7B-Instruct", # TODO implement SimplifiedLayerNormalization, RotaryEmbedding, MultiHeadAttention

  # ran out of memory on m1 mac
  "stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/sdxl-turbo", "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
  "distil-whisper/distil-large-v2", "distil-whisper/distil-large-v3",
  "Snowflake/snowflake-arctic-embed-m-v2.0",

  # TODO MOD bug with const folding
  # There's a huge concat in here with 1024 shape=(1, 3, 32, 32) Tensors
  "briaai/RMBG-2.0"
]

def huggingface_download_onnx_model(model_id:str):
  return Path(snapshot_download(repo_id=model_id, allow_patterns=["*.onnx", "*.onnx_data", "*config.json"], cache_dir=_ensure_downloads_dir()))

def get_config(root_path:Path):
  config_paths = list(root_path.rglob("config.json")) + list(root_path.rglob("preprocessor_config.json"))
  return {k:v for path in config_paths for k,v in json.load(path.open()).items()}

def get_tolerances(file_name):
  # TODO very high rtol atol
  # if "fp16" in file_name: return 9e-2, 9e-2
  # if any(q in file_name for q in ["int8", "uint8", "quantized"]): return 4, 4
  return 3e-3, 3e-3

def run_huggingface_benchmark(onnx_model_path, config, rtol, atol):
  inputs = get_example_inputs(OnnxRunner(onnx.load(onnx_model_path)).graph_inputs, config)
  validate(onnx_model_path, inputs, rtol=rtol, atol=atol)

if __name__ == "__main__":
  limit = getenv("LIMIT")
  repo_path = getenv("REPOPATH", "")
  model_path = getenv("MODELPATH", "")
  assert limit or repo_path or model_path, \
    """
    Please provide one of these environment variables:
    - 'LIMIT=100' (to run top N models)
    - 'REPOPATH=google-bert/bert-base-uncased' (to debug all onnx models inside repo)
    - 'MODELPATH=google-bert/bert-base-uncased/model.onnx' (to debug a single model)
      - optionally use 'TRUNCATE=50' with 'MODELPATH' to test intermediate results"""

  # for running
  if limit:
    sort = "downloads"  # recent 30 days downloads
    result = {"passed": 0, "failed": 0}
    print(f"** Running benchmarks on top {limit} models ranked by '{sort}' on huggingface **")
    for i, model in enumerate(list_models(filter="onnx", sort=sort, limit=limit)):
      if model.id in SKIPPED_REPO_PATHS: continue  # skip these
      print(f"{i}: ({getattr(model, sort)} {sort}) ")
      url = f"{HUGGINGFACE_URL}/{model.id}"
      result[model.id] = {"url": url}
      print(f"Downloading all onnx models from {url}")
      root_path = huggingface_download_onnx_model(model.id)
      print(f"Saved to {root_path}")
      for onnx_model_path in root_path.rglob("*.onnx"):
        onnx_file_name = onnx_model_path.stem
        if any(skip in onnx_file_name for skip in SKIPPED_FILES): continue  # skip these
        rtol, atol = get_tolerances(onnx_file_name)
        relative_path = str(onnx_model_path.relative_to(root_path))
        print(f"Benchmarking {relative_path}")
        try:
          run_huggingface_benchmark(onnx_model_path, get_config(root_path), rtol, atol)
          result[model.id][relative_path] = {"status": "passed"}
          result["passed"] += 1
        except Exception as e:
          result[model.id][relative_path] = {"status": f"failed {e}"}
          result["failed"] += 1

    with open("huggingface_results.json", "w") as f:
      json.dump(result, f, indent=2)
      print(f"report saved to {Path('huggingface_results.json').resolve()}")

  # for debugging
  # `repo_path` is `model.id`
  if repo_path:
    root_path = huggingface_download_onnx_model(repo_path)
    for onnx_model_path in root_path.rglob("*.onnx"):
      rtol, atol = get_tolerances(onnx_model_path.stem)
      relative_path = str(onnx_model_path.relative_to(root_path))
      try:
        run_huggingface_benchmark(onnx_model_path, get_config(root_path), rtol, atol)
        print(f"{relative_path} passed")
      except Exception as e:
        print(f"{relative_path} failed")
        print(e)

  # for debugging
  # `model_path` is `model.id + relative_path`
  if model_path:
    model_id, relative_path = model_path.split("/", 2)[:2], model_path.split("/", 2)[2]
    onnx_file_name = model_id[-1]
    rtol, atol = get_tolerances(onnx_file_name)
    model_id = "/".join(model_id)
    root_path = huggingface_download_onnx_model(model_id)
    onnx_model = root_path / relative_path
    if (limit := getenv("TRUNCATE", -1)) != -1:
      # truncates the onnx model so intermediate results can be validated
      model = onnx.load(onnx_model)
      nodes_up_to_limit = list(model.graph.node)[:limit+1]
      new_output_values = [onnx.helper.make_empty_tensor_value_info(output_name) for output_name in nodes_up_to_limit[-1].output]
      model.graph.ClearField("node")
      model.graph.node.extend(nodes_up_to_limit)
      model.graph.ClearField("output")
      model.graph.output.extend(new_output_values)
      with tempfile.NamedTemporaryFile(suffix=onnx_model.suffix) as tmp:
        onnx.save(model, tmp.name)
        run_huggingface_benchmark(tmp.name, get_config(root_path), rtol, atol)
    else:
      run_huggingface_benchmark(onnx_model, get_config(root_path), rtol, atol)