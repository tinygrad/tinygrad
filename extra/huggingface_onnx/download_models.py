import yaml, argparse, time
from requests.exceptions import ConnectionError, ReadTimeout, Timeout
from pathlib import Path
from huggingface_hub import snapshot_download, errors
import concurrent.futures

def snapshot_with_retries(model_id, allow_patterns, cache_dir, retries=3, timeout=5):
    for _ in range(retries):
        try:
            snapshot_path = snapshot_download(repo_id=model_id, allow_patterns=allow_patterns, cache_dir=cache_dir)
            return snapshot_path
        except (ConnectionError, ReadTimeout, errors.LocalEntryNotFoundError) as e:
            exception = e
            print(f"Encountered timeout while downloading `{model_id}` with patterns {allow_patterns}, retrying.")
            time.sleep(timeout)
    raise Timeout(
        f"Failed to download model '{model_id}' with patterns {allow_patterns} after {retries} retries."
    ) from exception

def fetch_model(model_id, model_data, download_dir):
    """Helper function to download a single model (with retries) and return the root path."""
    allow_patterns = [file_info["file"] for file_info in model_data["files"]]
    # Download config files as well
    allow_patterns += ["*config.json"]

    print(f"Downloading: {model_id} with patterns: {allow_patterns}")
    root_path = Path(snapshot_with_retries(model_id, allow_patterns, download_dir))
    print(f"Downloaded {model_id} files to: {root_path}")
    return model_id, root_path

def download_models(yaml_file: str, download_dir: str, max_workers=4) -> None:
    """Download multiple models from Hugging Face in parallel."""
    # Load metadata
    with open(yaml_file, 'r') as f:
        metadata = yaml.safe_load(f)

    # Prepare concurrency
    repos = metadata["repositories"]
    print(f"Starting parallel download for {len(repos)} repositories...")

    # Submit all downloads to a ThreadPool
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(fetch_model, model_id, model_data, download_dir): model_id
            for model_id, model_data in repos.items()
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_model):
            model_id = future_to_model[future]
            try:
                _, root_path = future.result()
                # Store the path back in metadata
                repos[model_id]["download_path"] = str(root_path)
            except Exception as exc:
                print(f"[ERROR] {model_id} generated an exception: {exc}")
                raise

    # Save the updated metadata
    with open(yaml_file, 'w') as f:
        yaml.dump(metadata, f, sort_keys=False)
    print("All downloads completed, YAML updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from HuggingFace Hub in parallel.")
    parser.add_argument("input", type=str, help="Path to the input YAML configuration file containing model information.")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of parallel threads to use.")
    args = parser.parse_args()

    models_folder = Path(__file__).parent / "models"
    models_folder.mkdir(parents=True, exist_ok=True)

    download_models(args.input, str(models_folder), max_workers=args.max_workers)
