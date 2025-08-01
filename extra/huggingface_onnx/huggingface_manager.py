import yaml
import time
import requests
import argparse
from dataclasses import asdict
from pathlib import Path
from huggingface_hub import list_models, HfApi, snapshot_download
from tinygrad.helpers import tqdm
from extra.onnx import OnnxRunner

# Constants for filtering models
HUGGINGFACE_URL = "https://huggingface.co"
SKIPPED_FILES = [
  "fp16", "int8", "uint8", "quantized",      # numerical accuracy issues
  "avx2", "arm64", "avx512", "avx512_vnni",  # numerical accuracy issues
  "q4", "q4f16", "bnb4",                     # unimplemented quantization
  "model_O4",                                # requires non cpu ort runner and MemcpyFromHost op
  "merged",                                  # TODO implement attribute with graph type and Loop op
]

SKIPPED_REPO_PATHS = [
  # Invalid model-index
  "AdamCodd/vit-base-nsfw-detector",
  # TODO: implement attribute with graph type and Loop op
  "minishlab/potion-base-8M", "minishlab/M2V_base_output", "minishlab/potion-retrieval-32M",
  # TODO: implement SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization, GroupQueryAttention
  "HuggingFaceTB/SmolLM2-360M-Instruct",
  # TODO: implement SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization, RotaryEmbedding, MultiHeadAttention
  "HuggingFaceTB/SmolLM2-1.7B-Instruct",
  # TODO: implement RandomNormalLike
  "stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/sdxl-turbo", 'SimianLuo/LCM_Dreamshaper_v7',
  # TODO: implement NonZero
  "mangoapps/fb_zeroshot_mnli_onnx",
  # TODO huge Concat in here with 1024 (1, 3, 32, 32) Tensors, and maybe a MOD bug with const folding
  "briaai/RMBG-2.0",
]


class HuggingFaceONNXManager:
  """Manages HuggingFace ONNX model discovery, metadata collection, and downloading."""

  def __init__(self, base_dir: Path = None):
    self.base_dir = base_dir or Path(__file__).parent
    self.models_dir = self.base_dir / "models"
    print(self.models_dir)
    self.api = HfApi()

  def discover_models(self, limit: int, sort: str = "downloads") -> list[str]:
    print(f"Discovering top {limit} ONNX models sorted by {sort}...")
    repos = []
    i = 0

    for model in list_models(filter="onnx", sort=sort):
      if model.id in SKIPPED_REPO_PATHS:
        continue

      print(f"  {i+1}/{limit}: {model.id} ({getattr(model, sort)})")
      repos.append(model.id)
      i += 1
      if i == limit:
        break

    print(f"Found {len(repos)} suitable ONNX models")
    return repos

  def collect_metadata(self, repos: list[str]) -> dict:
    print(f"Collecting metadata for {len(repos)} repositories...")
    metadata = {"repositories": {}}
    total_size = 0

    for repo in tqdm(repos, desc="Collecting metadata"):
      try:
        files_metadata = []
        model_info = self.api.model_info(repo)

        for file in model_info.siblings:
          filename = file.rfilename
          if not (filename.endswith('.onnx') or filename.endswith('.onnx_data')):
            continue
          if any(skip_str in filename for skip_str in SKIPPED_FILES):
            continue

          # Get file size from API or HEAD request
          try:
            head = requests.head(
              f"{HUGGINGFACE_URL}/{repo}/resolve/main/{filename}",
              allow_redirects=True,
              timeout=10
            )
            file_size = file.size or int(head.headers.get('Content-Length', 0))
          except requests.RequestException:
            file_size = file.size or 0

          files_metadata.append({
            "file": filename,
            "size": f"{file_size/1e6:.2f}MB"
          })
          total_size += file_size

        if files_metadata:  # Only add repos with valid ONNX files
          metadata["repositories"][repo] = {
            "url": f"{HUGGINGFACE_URL}/{repo}",
            "download_path": None,
            "files": files_metadata,
          }

      except Exception as e:
        print(f"WARNING: Failed to collect metadata for {repo}: {e}")
        continue

    metadata['total_size'] = f"{total_size/1e9:.2f}GB"
    metadata['created_at'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    print(f"Collected metadata for {len(metadata['repositories'])} repositories")
    print(f"Total estimated download size: {metadata['total_size']}")

    return metadata

  def download_models(self, metadata: dict, download_to_models_dir: bool = True) -> dict:
    download_dir = self.models_dir if download_to_models_dir else self.base_dir
    download_dir.mkdir(parents=True, exist_ok=True)

    repos = metadata["repositories"]
    n = len(repos)

    print(f"Downloading {n} repositories to {download_dir}...")

    for i, (model_id, model_data) in enumerate(repos.items()):
      print(f"  Downloading {i+1}/{n}: {model_id}...")

      try:
        # Download ONNX model files
        allow_patterns = [file_info["file"] for file_info in model_data["files"]]
        root_path = Path(snapshot_download(
          repo_id=model_id,
          allow_patterns=allow_patterns,
          cache_dir=str(download_dir)
        ))

        # Download config files (usually small)
        snapshot_download(
          repo_id=model_id,
          allow_patterns=["*config.json"],
          cache_dir=str(download_dir)
        )

        model_data["download_path"] = str(root_path)
        print(f"    Downloaded to: {root_path}")

        # Extract graph inputs for each ONNX file
        try:
          for file_info in model_data["files"]:
            if file_info["file"].endswith(".onnx"):
              onnx_path = root_path / file_info["file"]
              if onnx_path.exists():
                try:
                  runner = OnnxRunner(str(onnx_path))
                  # Convert graph_inputs to serializable format
                  graph_inputs = {}
                  for name, spec in runner.graph_inputs.items():
                    spec_dict = asdict(spec)
                    spec_dict["shape"] = list(spec_dict["shape"])
                    spec_dict["dtype"] = spec_dict["dtype"]['name']
                    graph_inputs[name] = spec_dict
                  file_info["graph_inputs"] = graph_inputs
                  print(f"      Extracted graph inputs for {file_info['file']}: {list(graph_inputs.keys())}")
                except Exception as input_error:
                  print(f"      WARNING: Failed to extract graph inputs for {file_info['file']}: {input_error}")
                  file_info["graph_inputs"] = None
        except Exception as graph_error:
          print(f"    WARNING: Error processing graph inputs for {model_id}: {graph_error}")

      except Exception as e:
        print(f"    ERROR: Failed to download {model_id}: {e}")
        model_data["download_path"] = None
        continue

    successful_downloads = sum(1 for repo in repos.values() if repo["download_path"] is not None)
    print(f"Successfully downloaded {successful_downloads}/{n} repositories")

    return metadata

  def save_metadata(self, metadata: dict, output_file: str):
    yaml_path = self.base_dir / output_file
    with open(yaml_path, 'w') as f:
      yaml.dump(metadata, f, sort_keys=False)
    print(f"Metadata saved to: {yaml_path}")

  def load_metadata(self, yaml_file: str) -> dict:
    yaml_path = self.base_dir / yaml_file
    with open(yaml_path, 'r') as f:
      return yaml.safe_load(f)

  def discover_and_download(self, limit: int, output_file: str = "huggingface_repos.yaml",
                          sort: str = "downloads", download: bool = True):
    print(f"Starting HuggingFace ONNX workflow...")
    print(f"   Limit: {limit} models")
    print(f"   Sort by: {sort}")
    print(f"   Download: {'Yes' if download else 'No'}")
    print(f"   Output: {output_file}")
    print("-" * 50)

    repos = self.discover_models(limit, sort)

    metadata = self.collect_metadata(repos)

    if download:
      metadata = self.download_models(metadata)

    self.save_metadata(metadata, output_file)

    print("-" * 50)
    print("Workflow completed successfully!")
    if download:
      successful = sum(1 for repo in metadata["repositories"].values()
                     if repo["download_path"] is not None)
      print(f"{successful}/{len(metadata['repositories'])} models downloaded")

    return metadata


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="HuggingFace ONNX Model Manager - Discover, collect metadata, and download ONNX models",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Discover and download top 50 models
  python huggingface_manager.py --limit 50 --download

  # Just collect metadata for top 100 models (no download)
  python huggingface_manager.py --limit 100

  # Sort by likes instead of downloads
  python huggingface_manager.py --limit 20 --sort likes --download
        """
  )

  parser.add_argument("--limit", type=int, help="Number of top repositories to process")
  parser.add_argument("--output", type=str, default="huggingface_repos.yaml",
                     help="Output YAML file name (default: huggingface_repos.yaml)")
  parser.add_argument("--sort", type=str, default="downloads",
                     choices=["downloads", "likes", "created", "modified"],
                     help="Sort criteria for model discovery (default: downloads)")

  parser.add_argument("--download", action="store_true", default=False,
                     help="Download models after collecting metadata")

  args = parser.parse_args()

  if not args.limit: parser.error("--limit is required")

  manager = HuggingFaceONNXManager()

  try:
    # Full workflow: discover and optionally download
    should_download = args.download
    manager.discover_and_download(
      limit=args.limit,
      output_file=args.output,
      sort=args.sort,
      download=should_download
    )

  except KeyboardInterrupt:
    print("\nProcess interrupted by user")
  except Exception as e:
    print(f"ERROR: {e}")
    exit(1)