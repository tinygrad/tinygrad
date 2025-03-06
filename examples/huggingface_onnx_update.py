import yaml, json, shutil, time
from pathlib import Path
from huggingface_hub import list_models, snapshot_download
from tinygrad.helpers import _ensure_downloads_dir, getenv

DOWNLOAD_DIR = _ensure_downloads_dir() / "huggingface_onnx"
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
  "distil-whisper/distil-large-v2", "distil-whisper/distil-large-v3", "SimianLuo/LCM_Dreamshaper_v7", "openai-community/gpt2-large",

  # TODO MOD bug with const folding
  # There's a huge concat in here with 1024 shape=(1, 3, 32, 32) Tensors
  "briaai/RMBG-2.0",

  # invalid model index
  "NTQAI/pedestrian_gender_recognition"
]

def download_repo_onnx_models(model_id:str):
  return Path(snapshot_download(repo_id=model_id, allow_patterns=["*.onnx", "*.onnx_data"], ignore_patterns=["*"+n+"*" for n in SKIPPED_FILES],
                                cache_dir=DOWNLOAD_DIR))

def download_repo_configs(model_id:str):
  return Path(snapshot_download(repo_id=model_id, allow_patterns=["*config.json"], cache_dir=DOWNLOAD_DIR))

def delete_all():
  for folder in DOWNLOAD_DIR.iterdir(): shutil.rmtree(folder)

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
  print(f"** Downloading top {n} repos sorted by {sort} (filter architectures: {filter_architecture}) **")
  for model in list_models(filter="onnx", sort=sort):
    if model.id in SKIPPED_REPO_PATHS: continue  # skip these
    architecture = get_config(download_repo_configs(model.id)).get('architectures', ["unknown"])[0]
    if filter_architecture and architecture in architecture_tracker and architecture != "unknown": continue # skip duplicated architectures
    architecture_tracker.add(architecture)

    print(f"{i+1}/{n}: {model.id} ({architecture} architecture)")
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

if __name__ == "__main__":
  limit = getenv("LIMIT")
  filter_architecture = getenv("FILTER_ARCHITECTURE")
  assert limit, \
    """
    - 'LIMIT=100' (to update top N repos and write to "huggingface_repos.yaml" which is used as the official list of models to test)
      - optionally use 'FILTER_ARCHITECTURE=1' to ensure each repo has a unique architecture"""
  # update huggingface_repos.yaml
  repos = download_top_repos(limit, filter_architecture, "downloads")
  with open('huggingface_repos.yaml', 'w') as f:
    repos.update({"last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    yaml.dump(repos, f, sort_keys=False)
