import onnx, yaml, json, tempfile, time, collections, gc, pprint
from pathlib import Path
from huggingface_hub import list_models, snapshot_download
from tinygrad.helpers import _ensure_downloads_dir, getenv
from extra.onnx import OnnxRunner, get_onnx_ops
from extra.onnx_helpers import validate, get_example_inputs

DOWNLOAD_DIR = _ensure_downloads_dir()
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
  "minishlab/potion-base-8M", "minishlab/M2V_base_output", "minishlab/potion-retrieval-32M", # TODO: implement attribute with graph type
  "HuggingFaceTB/SmolLM2-360M-Instruct", # TODO implement GroupQueryAttention
  "HuggingFaceTB/SmolLM2-1.7B-Instruct", # TODO implement SimplifiedLayerNormalization, RotaryEmbedding, MultiHeadAttention
  "segment-any-text/sat-12l-sm", # TODO: true float16

  # ran out of memory on m1 mac
  "stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/sdxl-turbo", "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
  "distil-whisper/distil-large-v2", "distil-whisper/distil-large-v3",

  # TODO MOD bug with const folding
  # There's a huge concat in here with 1024 shape=(1, 3, 32, 32) Tensors
  "briaai/RMBG-2.0",

  # invalid model index
  "NTQAI/pedestrian_gender_recognition"
]

def run_huggingface_validate(onnx_model_path, config, rtol, atol):
  try:
    onnx_model = onnx.load(onnx_model_path)
    onnx_runner = OnnxRunner(onnx_model)
    inputs = get_example_inputs(onnx_runner.graph_inputs, config)
    validate(onnx_model_path, inputs, rtol=rtol, atol=atol)
  finally:
    del onnx_model
    del onnx_runner
    del inputs
    gc.collect()

def download_repo_onnx_models(model_id:str):
  return Path(snapshot_download(repo_id=model_id, allow_patterns=["*.onnx", "*.onnx_data"], ignore_patterns=["*"+n+"*" for n in SKIPPED_FILES],
                                cache_dir=DOWNLOAD_DIR))

def download_repo_configs(model_id:str):
  return Path(snapshot_download(repo_id=model_id, allow_patterns=["*config.json"], cache_dir=DOWNLOAD_DIR))

def get_config(root_path:Path):
  ret = {}
  for path in root_path.rglob("*config.json"):
    config = json.load(path.open())
    if isinstance(config, dict): ret.update(config)
  return ret

def download_top_repos(n:int, filter_architecture:bool, sort="downloads") -> dict:
  ret = {"repositories": {}}
  architecture_tracker = set()
  i = 0
  print(f"Downloading top {n} repos sorted by {sort} (filter architectures: {filter_architecture})")
  for model in list_models(filter="onnx", sort=sort):
    if model.id in SKIPPED_REPO_PATHS: continue  # skip these
    architecture = get_config(download_repo_configs(model.id)).get('architectures', ["unknown"])[0]
    if filter_architecture and architecture in architecture_tracker and architecture != "unknown": continue # skip duplicated architectures
    architecture_tracker.add(architecture)

    print(f"Processing repo {i+1}/{n}: {model.id} ({architecture} architecture)")
    print(f"Downloading ONNX models for {model.id}...")
    root_path = download_repo_onnx_models(model.id)
    print(f"Models downloaded to: {root_path}")

    model_metadata = []
    for onnx_model_path in root_path.rglob("*.onnx"):
      model_metadata.append({
        "model": str(onnx_model_path.relative_to(root_path)),
        "size": f"{onnx_model_path.stat().st_size/1e6:.2f}MB",
      })
    ret["repositories"][model.id] = {
      "30 day downloads": getattr(model, sort),
      "architecture": architecture,
      "url": f"{HUGGINGFACE_URL}/{model.id}",
      "path": str(root_path),
      "models": model_metadata
    }
    i += 1
    if i == n: break
  return ret

def get_tolerances(file_name):
  # TODO very high rtol atol
  # if "fp16" in file_name: return 9e-2, 9e-2
  # if any(q in file_name for q in ["int8", "uint8", "quantized"]): return 4, 4
  return 4e-3, 4e-3

def validate_repos(models:dict[str, tuple[Path, Path]]):
  print(f"** Validating {len(model_paths)} repos **")
  for model_id, (root_path, relative_path) in models.items():
    print(f"validating model {model_id}")
    model_path = root_path / relative_path
    onnx_file_name = model_path.stem
    config = get_config(root_path)
    rtol, atol = get_tolerances(onnx_file_name)
    st = time.time()
    run_huggingface_validate(model_path, config, rtol, atol)
    et = time.time() - st
    print(f"passed, took {et:.2f}s")

def retrieve_op_stats(models:dict[str, tuple[Path, Path]]) -> dict:
  ret = {}
  op_counter = collections.Counter()
  unsupported_ops = collections.defaultdict(set)
  supported_ops = get_onnx_ops()
  print(f"** Retrieving stats for {len(model_paths)} models **")
  for model_id, (root_path, relative_path) in models.items():
    model_path = root_path / relative_path
    onnx_runner = OnnxRunner(onnx.load(model_path))
    for node in onnx_runner.graph_nodes:
      op_counter[node.op] += 1
      if node.op not in supported_ops:
        unsupported_ops[node.op].add(model_id)
    del onnx_runner
  ret["unsupported_ops"] = {k:list(v) for k, v in unsupported_ops.items()}
  ret["op_counter"] = op_counter.most_common()
  return ret

if __name__ == "__main__":
  update_limit = getenv("UPDATE")
  filter_architecture = getenv("FILTER_ARCHITECTURE")
  do_check = getenv("CHECK_OPS")
  do_validate = getenv("VALIDATE")
  debug_model_path = getenv("MODELPATH", "")
  truncate = getenv("TRUNCATE", -1)
  assert update_limit or do_check or do_validate or debug_model_path, \
    """
    Please provide either environment variable `UPDATE`, `CHECK_OPS`, `VALIDATE`, or `MODELPATH`:
    - 'UPDATE=100' (to update top N repos and write to "huggingface_repos.yaml" which is used as the official list of models to test)
      - optionally use 'FILTER_ARCHITECTURE=1' to ensure each repo has a unique architecture
    - 'CHECK_OPS=1' (to check ops for repos in "huggingface_repos.yaml" and print out a report)
    - 'VALIDATE=1' (to validate correctness for repos in "huggingface_repos.yaml")
    - 'MODELPATH=google-bert/bert-base-uncased/model.onnx' (to debug run validation on a single model)
      - optionally use 'TRUNCATE=50' with 'MODELPATH' to validate intermediate results"""

  # for running
  # default values for running for the first time
  if not Path("huggingface_repos.yaml").exists():
    update_limit = 80
    filter_architecture = True

  # update huggingface_repos.yaml
  if update_limit:
    repos = download_top_repos(update_limit, filter_architecture, "downloads")
    with open('huggingface_repos.yaml', 'w') as f:
      repos.update({"last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
      yaml.dump(repos, f)

  if do_check or do_validate:
    with open('huggingface_repos.yaml', 'r') as f:
      data = yaml.safe_load(f)
      model_paths = {
        model_id + model["model"]: (Path(repo["path"]), Path(model["model"]))
        for model_id, repo in data["repositories"].items()
        for model in repo["models"]
      }

    if do_check:
      pprint.pprint(retrieve_op_stats(model_paths))

    if do_validate:
      validate_repos(model_paths)

  # for debugging
  # `model_path` is `model.id + relative_path` ("google-bert/bert-base-uncased/model.onnx")
  if debug_model_path:
    print(f"debugging with {debug_model_path=}")
    model_id, relative_path = debug_model_path.split("/", 2)[:2], debug_model_path.split("/", 2)[2]
    onnx_file_name = model_id[-1]
    rtol, atol = get_tolerances(onnx_file_name)
    model_id = "/".join(model_id)
    root_path = download_repo_onnx_models(model_id)
    config = download_repo_configs(model_id)
    onnx_model = root_path / relative_path
    if truncate != -1:
      # truncates the onnx model so intermediate results can be validated
      model = onnx.load(onnx_model)
      nodes_up_to_limit = list(model.graph.node)[:truncate+1]
      new_output_values = [onnx.helper.make_empty_tensor_value_info(output_name) for output_name in nodes_up_to_limit[-1].output]
      model.graph.ClearField("node")
      model.graph.node.extend(nodes_up_to_limit)
      model.graph.ClearField("output")
      model.graph.output.extend(new_output_values)
      with tempfile.NamedTemporaryFile(suffix=onnx_model.suffix) as tmp:
        onnx.save(model, tmp.name)
        run_huggingface_validate(tmp.name, config, rtol, atol)
    else:
      run_huggingface_validate(onnx_model, config, rtol, atol)