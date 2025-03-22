import yaml, argparse, time, asyncio
from requests.exceptions import ConnectionError, ReadTimeout, Timeout
from pathlib import Path
from huggingface_hub import snapshot_download, errors
import threading

class DummyTqdm:
  _lock = threading.Lock()
  @classmethod
  def set_lock(cls, lock): cls._lock = lock
  def __init__(self, iterable, **kwargs): self.iterable = iterable
  def __iter__(self): return iter(self.iterable)
  def update(self, n): pass
  def close(self): pass

async def snapshot_with_retries(model_id, allow_patterns, cache_dir, retries=3, timeout=5):
  for _ in range(retries):
    try:
      # Run snapshot_download in a separate thread
      snapshot_path = await asyncio.to_thread(
        snapshot_download, repo_id=model_id, allow_patterns=allow_patterns, cache_dir=cache_dir, tqdm_class=DummyTqdm
      )
      return snapshot_path
    except (ConnectionError, ReadTimeout, errors.LocalEntryNotFoundError) as e:
      exception = e
      print(f"Encountered timeout while downloading `{model_id}` with patterns {allow_patterns}, retrying.")
      await asyncio.sleep(timeout)
  raise Timeout(f"Failed to download model '{model_id}' with patterns {allow_patterns} after {retries} retries. Aborting CI process.") from exception

async def fetch_model(model_id, model_data, download_dir):
  """Helper function to download a single model (with retries) and return the root path."""
  allow_patterns = [file_info["file"] for file_info in model_data["files"]] + ["*config.json"]
  print(f"Downloading: {model_id} with patterns: {allow_patterns}")
  root_path = Path(await snapshot_with_retries(model_id, allow_patterns, download_dir))
  print(f"Downloaded {model_id} files to: {root_path}")
  return model_id, root_path

async def download_models(yaml_file: str, download_dir: str) -> None:
  """Download multiple models from Hugging Face in parallel asynchronously."""
  # Load metadata
  with open(yaml_file, 'r') as f: metadata = yaml.safe_load(f)
  repos = metadata["repositories"]

  print(f"Starting parallel download for {len(repos)} repositories...")
  tasks = [fetch_model(model_id, model_data, download_dir) for model_id, model_data in repos.items()]
  results = await asyncio.gather(*tasks, return_exceptions=False)

  for model_id, root_path in results: repos[model_id]["download_path"] = str(root_path)
  with open(yaml_file, 'w') as f: yaml.dump(metadata, f, sort_keys=False)
  print("All downloads completed, YAML updated.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Download models from Huggingface Hub based on a YAML configuration file.")
  parser.add_argument("input", type=str, help="Path to the input YAML configuration file containing model information.")
  args = parser.parse_args()

  models_folder = Path(__file__).parent / "models"
  models_folder.mkdir(parents=True, exist_ok=True)

  asyncio.run(download_models(args.input, str(models_folder)))
