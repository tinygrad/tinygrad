import yaml, json, time, requests, argparse
from pathlib import Path
from huggingface_hub import list_models, snapshot_download, HfApi
from tinygrad.helpers import _ensure_downloads_dir

HUGGINGFACE_URL = "https://huggingface.co"
SKIPPED_FILES = [
  "fp16", "int8", "uint8", "quantized",      # numerical accuracy issues
  "avx2", "arm64", "avx512", "avx512_vnni",  # numerical accuracy issues
  "q4", "q4f16", "bnb4",                     # unimplemented quantization
  "model_O4",                                # requires non cpu ort runner and MemcpyFromHost op
  "merged",                                  # TODO implement attribute with graph type
]
SKIPPED_REPO_PATHS = [
  # TODO: implement attribute with graph type
  "minishlab/potion-base-8M", "minishlab/M2V_base_output", "minishlab/potion-retrieval-32M",
  # TODO: implement SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization, GroupQueryAttention
  "HuggingFaceTB/SmolLM2-360M-Instruct",
  # TODO: implement SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization, RotaryEmbedding, MultiHeadAttention
  "HuggingFaceTB/SmolLM2-1.7B-Instruct",
  # TODO: implmement RandomNormalLike
  "stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/sdxl-turbo", 'SimianLuo/LCM_Dreamshaper_v7',
  # TODO: implement NonZero
  "mangoapps/fb_zeroshot_mnli_onnx",
  # TODO huge Concat in here with 1024 (1, 3, 32, 32) Tensors, and maybe a MOD bug with const folding
  "briaai/RMBG-2.0",
]

def download_repo_onnx_models(model_id: str, download_dir:str):
  return Path(snapshot_download(repo_id=model_id, allow_patterns=["*.onnx", "*.onnx_data"], ignore_patterns=["*" + n + "*" for n in SKIPPED_FILES],
                                cache_dir=download_dir))

def download_repo_configs(model_id: str, download_dir:str):
  return Path(snapshot_download(repo_id=model_id, allow_patterns=["*config.json"], cache_dir=download_dir))

def get_config(root_path: Path):
  ret = {}
  for path in root_path.rglob("*config.json"):
    config = json.load(path.open())
    if isinstance(config, dict):
      ret.update(config)
  return ret

def download_top_repos(n:int, filter_architecture:bool, sort:str, dry_run:bool, download_dir:str) -> dict:
  api = HfApi()
  ret = {"repositories": {}}
  architecture_tracker = set()
  total_size = 0
  i = 0
  log_prefix = 'Getting metadata for' if dry_run else 'Downloading'
  print(f"** {log_prefix} top {n} repos {sort=}, {filter_architecture=}, {dry_run=}, {download_dir=} **")
  for model in list_models(filter="onnx", sort=sort):
    if model.id in SKIPPED_REPO_PATHS: continue  # skip these
    # always download the configs (small files) to get the architecture
    architecture = get_config(download_repo_configs(model.id, download_dir)).get('architectures', ["unknown"])[0]
    if filter_architecture and architecture in architecture_tracker and architecture != "unknown": continue  # skip duplicated architectures
    architecture_tracker.add(architecture)

    print(f"{i+1}/{n}: {model.id} ({architecture} architecture)")
    files_metadata = []
    if dry_run:
      print(f"Dry run: getting metadata of ONNX models for {model.id}...")
      model_info = api.model_info(model.id)
      for file in model_info.siblings:
        filename = file.rfilename
        if not (filename.endswith('.onnx') or filename.endswith('.onnx_data')): continue
        if any(skip_str in filename for skip_str in SKIPPED_FILES): continue
        file_size = file.size
        if file_size is None:
          file_url = f"{HUGGINGFACE_URL}/{model.id}/resolve/main/{filename}"
          head = requests.head(file_url, allow_redirects=True)
          file_size = int(head.headers.get('Content-Length'))
        files_metadata.append({
          "file": filename,
          "size": f"{file_size/1e6:.2f}MB",
        })
        total_size += file_size
    else:
      print(f"Real run: downloading ONNX models for {model.id}...")
      root_path = download_repo_onnx_models(model.id, download_dir)
      print(f"Models downloaded to: {root_path}")
      files = list(root_path.rglob("*.onnx")) + list(root_path.rglob("*.onnx_data"))
      for fp in files:
        file_size = fp.stat().st_size
        files_metadata.append({
          "file": str(fp.relative_to(root_path)),
          "size": f"{file_size/1e6:.2f}MB",
        })
        total_size += file_size

    ret["repositories"][model.id] = {
      "30 day downloads": getattr(model, sort),
      "architecture": architecture,
      "url": f"{HUGGINGFACE_URL}/{model.id}",
      "path": None if dry_run else str(root_path),
      "files": files_metadata
    }
    i += 1
    if i == n:
      break
  ret['total_size'] = f"{total_size/1e9:.2f}GB"
  return ret

if __name__ == "__main__":
  sort = "downloads" # recent 30 days downloads
  DOWNLOAD_DIR = _ensure_downloads_dir() / "huggingface_onnx"
  parser = argparse.ArgumentParser(description="Download or get metadata of top Huggingface ONNX repositories")
  parser.add_argument("--limit", type=int, required=True, help="Number of top repositories to process (e.g., 100)")
  parser.add_argument("--filter_architecture", action="store_true", default=False, help="Ensure each repository has a unique architecture")
  parser.add_argument("--dry_run", action="store_true", default=False, help="Get metadata without downloading the models")
  parser.add_argument("--output_file", type=str, default="huggingface_repos.yaml", help="Output YAML file name to save the report")
  parser.add_argument("--download_dir", type=str, default=str(DOWNLOAD_DIR), help="Directory to download repositories to (default: %(default)s)")
  args = parser.parse_args()

  repos = download_top_repos(args.limit, args.filter_architecture, sort, args.dry_run, args.download_dir)
  with open(args.output_file, 'w') as f:
    repos.update({"last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    yaml.dump(repos, f, sort_keys=False)
  print(f"YAML saved to: {args.output_file}")