import onnx, json, tempfile, time, collections
from pathlib import Path
from huggingface_hub import list_models, snapshot_download
from tinygrad.helpers import _ensure_downloads_dir, getenv
from extra.onnx import OnnxRunner, get_onnx_ops
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
  "minishlab/potion-base-8M", "minishlab/M2V_base_output", "minishlab/potion-retrieval-32M" # implement attribute with graph type
  "HuggingFaceTB/SmolLM2-360M-Instruct", # TODO implement GroupQueryAttention
  "HuggingFaceTB/SmolLM2-1.7B-Instruct", # TODO implement SimplifiedLayerNormalization, RotaryEmbedding, MultiHeadAttention
  "segment-any-text/sat-12l-sm", # TODO: true float16

  # ran out of memory on m1 mac
  "stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/sdxl-turbo", "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
  "distil-whisper/distil-large-v2", "distil-whisper/distil-large-v3", "bigscience/bloom-560m", "bigscience/bloom-1b1",
  "Snowflake/snowflake-arctic-embed-m-v2.0",

  # TODO MOD bug with const folding
  # There's a huge concat in here with 1024 shape=(1, 3, 32, 32) Tensors
  "briaai/RMBG-2.0",

  # invalid model index
  "NTQAI/pedestrian_gender_recognition"
]

def download_repo_onnx_models(model_id:str):
  return Path(snapshot_download(repo_id=model_id, allow_patterns=["*.onnx", "*.onnx_data"], cache_dir=_ensure_downloads_dir()))

def download_repo_configs(model_id:str):
  root_path = Path(snapshot_download(repo_id=model_id, allow_patterns=["*config.json"], cache_dir=_ensure_downloads_dir()))
  ret = {}
  for path in root_path.rglob("*config.json"):
    config = json.load(path.open())
    if isinstance(config, dict): ret.update(config)
  return ret

def get_tolerances(file_name):
  # TODO very high rtol atol
  # if "fp16" in file_name: return 9e-2, 9e-2
  # if any(q in file_name for q in ["int8", "uint8", "quantized"]): return 4, 4
  return 4e-3, 4e-3

def run_huggingface_benchmark(onnx_model_path, config, rtol, atol):
  inputs = get_example_inputs(OnnxRunner(onnx.load(onnx_model_path)).graph_inputs, config)
  validate(onnx_model_path, inputs, rtol=rtol, atol=atol)

def top_models_validate(n:int, filter_architecture:bool, sort="downloads") -> dict:
  ret = {"passed": 0, "failed": 0}
  start_time = time.time()
  total_size_bytes = 0
  architectures = set()
  i = 1
  print(f"** Running benchmarks on top {n} models ranked by {sort=} with {filter_architecture=} **")
  for model in list_models(filter="onnx", sort=sort):
    # skip models
    if model.id in SKIPPED_REPO_PATHS: continue  # skip these
    config = download_repo_configs(model.id)
    architecture = config.get('architectures', ["unknown"])[0]
    if filter_architecture and architecture in architectures and architecture != "unknown":
      print(f"skipped {model.id}, architecture '{architecture}' already validated")
      continue
    ret[model.id] = {}
    ret[model.id]['architecture'] = architecture
    architectures.add(architecture)

    print(f"{i}: {model.id} ({getattr(model, sort)} {sort}) ")
    url = f"{HUGGINGFACE_URL}/{model.id}"
    ret[model.id] = {"url": url}
    print(f"Downloading all onnx models from {url}")
    root_path = download_repo_onnx_models(model.id)

    total_size = sum(f.stat().st_size for f in root_path.glob('**/*') if f.is_file())
    total_size_bytes += total_size
    ret[model.id]['total_size'] = f"{total_size / 1e9:.2f}GB"
    print(f"Saved {total_size/1e6:.2f}MB to {root_path}")

    for onnx_model_path in root_path.rglob("*.onnx"):
      onnx_file_name = onnx_model_path.stem
      if any(skip in onnx_file_name for skip in SKIPPED_FILES): continue  # skip these
      rtol, atol = get_tolerances(onnx_file_name)
      relative_path = str(onnx_model_path.relative_to(root_path))
      print(f"Benchmarking {relative_path}")
      try:
        st = time.time()
        run_huggingface_benchmark(onnx_model_path, config, rtol, atol)
        et = time.time() - st
        ret[model.id][relative_path] = {"status": "passed", "time": f"{et:.2f}s"}
        ret["passed"] += 1
      except Exception as e:
        ret[model.id][relative_path] = {"status": f"failed {e}"}
        ret["failed"] += 1

    i += 1
    if i == n: break

  total_seconds = time.time() - start_time
  ret["total_time"] = f"{int(total_seconds // 60)}m {total_seconds % 60:.2f}s"
  ret["total_size"] = f"{total_size_bytes / 1e9:.2f}GB"
  return ret

def top_models_stats(n:int, filter_architecture:bool, sort="downloads") -> dict:
  ret = {}
  op_counter = collections.Counter()
  architecture_counter = collections.Counter()
  unsupported_ops = collections.defaultdict(set)
  supported_ops = get_onnx_ops()
  i = 1
  print(f"** Retrieving stats for top {n} models ranked by {sort=} with {filter_architecture=} **")
  for model in list_models(filter="onnx", sort=sort):
    if model.id in SKIPPED_REPO_PATHS: continue  # skip repo
    config = download_repo_configs(model.id)
    architecture = config.get('architectures', ["unknown"])[0]
    if filter_architecture and architecture in architecture_counter and architecture != "unknown": continue # skip same architecture repo

    print(f"{i}: {model.id} ({getattr(model, sort)} {sort}) ")
    architecture_counter[architecture] += 1
    for onnx_model_path in download_repo_onnx_models(model.id).rglob("*.onnx"):
      if any(skip in onnx_model_path.stem for skip in SKIPPED_FILES): continue  # skip model
      for node in OnnxRunner(onnx.load(onnx_model_path)).graph_nodes:
        op_counter[node.op] += 1
        if node.op not in supported_ops:
          unsupported_ops[node.op].add(model.id)

    i += 1
    if i == n: break

  ret["unsupported_ops"] = unsupported_ops
  ret["op_counter"] = op_counter.most_common()
  ret["architecture_counter"] = architecture_counter.most_common()
  print(ret)
  return ret

if __name__ == "__main__":
  limit = getenv("LIMIT")
  repo_path = getenv("REPOPATH", "")
  model_path = getenv("MODELPATH", "")
  assert sum([bool(limit), bool(repo_path), bool(model_path)]) == 1, \
    """
    Please provide exactly ONE of the following environment variables:
    - 'LIMIT=100' (to run top N repos, saved to "huggingface_results.json")
      - optionally use `FILTER_ARCHITECTURE=1` to ensure each repo has a unique architecture
      - optionally use `CHECK_SUPPORT=1` to check onnx op support coverage instead of running validation
    - 'REPOPATH=google-bert/bert-base-uncased' (to debug all onnx models inside repo)
    - 'MODELPATH=google-bert/bert-base-uncased/model.onnx' (to debug a single model)
      - optionally use 'TRUNCATE=50' with 'MODELPATH' to test intermediate results"""

  # for running
  if limit:
    sort = "downloads"  # recent 30 days downloads
    filter_architecture = getenv("FILTER_ARCHITECTURE")
    if getenv("CHECK_SUPPORT"):
      result = top_models_stats(limit, filter_architecture, sort)
    else:
      result = top_models_validate(limit, filter_architecture, sort)
    with open("huggingface_results.json", "w") as f:
      json.dump(result, f, indent=2)
      print(f"report saved to {Path('huggingface_results.json').resolve()}")

  # for debugging
  # `repo_path` is `model.id`
  if repo_path:
    print(f"debugging with {repo_path=}")
    root_path = download_repo_onnx_models(repo_path)
    config = download_repo_configs(repo_path)
    for onnx_model_path in root_path.rglob("*.onnx"):
      rtol, atol = get_tolerances(onnx_model_path.stem)
      relative_path = str(onnx_model_path.relative_to(root_path))
      try:
        run_huggingface_benchmark(onnx_model_path, config, rtol, atol)
        print(f"{relative_path} passed")
      except Exception as e:
        print(f"{relative_path} failed")
        print(e)

  # for debugging
  # `model_path` is `model.id + relative_path`
  if model_path:
    print(f"debugging with {model_path=}")
    model_id, relative_path = model_path.split("/", 2)[:2], model_path.split("/", 2)[2]
    onnx_file_name = model_id[-1]
    rtol, atol = get_tolerances(onnx_file_name)
    model_id = "/".join(model_id)
    root_path = download_repo_onnx_models(model_id)
    config = download_repo_configs(model_id)
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
        run_huggingface_benchmark(tmp.name, config, rtol, atol)
    else:
      run_huggingface_benchmark(onnx_model, config, rtol, atol)