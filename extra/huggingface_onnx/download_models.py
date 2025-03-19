import yaml, argparse, time
from requests.exceptions import ConnectionError, ReadTimeout, Timeout
from pathlib import Path
from huggingface_hub import snapshot_download

def snapshot_with_retries(model_id, allow_patterns, cache_dir, retries=3, timeout=5):
  for _ in range(retries):
    try:
      snapshot_path = snapshot_download(repo_id=model_id, allow_patterns=allow_patterns, cache_dir=cache_dir)
      return snapshot_path
    except (ConnectionError, ReadTimeout) as e:
      exception = e
      print(f"Encountered timeout while downloading `{model_id}` with patterns {allow_patterns}, retrying.")
      time.sleep(timeout)
  raise Timeout( f"Failed to download model '{model_id}' with patterns {allow_patterns} after {retries} retries. Aborting CI process.") \
    from exception

def download_models(yaml_file: str, download_dir: str) -> None:
  with open(yaml_file, 'r') as f: metadata = yaml.safe_load(f)
  n = len(metadata["repositories"])

  for i, (model_id, model_data) in enumerate(metadata["repositories"].items()):
    print(f"Downloading {i+1}/{n}: {model_id}...")
    allow_patterns = [file_info["file"] for file_info in model_data["files"]]
    # download configs too (the sizes are small)
    allow_patterns = allow_patterns + ["*config.json"]
    root_path = Path(snapshot_with_retries(model_id, allow_patterns, download_dir))
    print(f"Downloaded model files to: {root_path}")
    model_data["download_path"] = str(root_path)

  # Save the updated metadata back to the YAML file
  with open(yaml_file, 'w') as f: yaml.dump(metadata, f, sort_keys=False)
  print("Download completed according to YAML file.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Download models from Huggingface Hub based on a YAML configuration file.")
  parser.add_argument("input", type=str, help="Path to the input YAML configuration file containing model information.")
  args = parser.parse_args()

  models_folder = Path(__file__).parent / "models"
  models_folder.mkdir(parents=True, exist_ok=True)
  download_models(args.input, str(models_folder))