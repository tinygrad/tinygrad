import json, sys, time
from pathlib import Path
from tinygrad.frontend.onnx import OnnxRunner, onnx_load
from extra.onnx_helpers import validate, get_example_inputs
from huggingface_hub import snapshot_download

def download_with_retry(repo_id, allow_patterns, cache_dir, retries=2, delay=3):
  for i in range(retries):
    try:
      return snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        cache_dir=str(cache_dir)
      )
    except Exception as e:
      if i == retries - 1:
        print(f"Download failed after {retries} attempts: {e}")
        raise
      print(f"Attempt {i+1} failed, retrying...")
      time.sleep(delay)

def get_config(root_path: Path) -> dict:
  ret = {}
  for path in root_path.rglob("*config.json"):
    config = json.load(path.open())
    if isinstance(config, dict):
      ret.update(config)
  return ret

def run_huggingface_validate(onnx_model_path: Path, config: dict, rtol: float, atol: float):
  onnx_model = onnx_load(onnx_model_path)
  onnx_runner = OnnxRunner(onnx_model)
  inputs = get_example_inputs(onnx_runner.graph_inputs, config)
  validate(onnx_model_path, inputs, rtol=rtol, atol=atol)

def get_tolerances(file_name: str) -> tuple[float, float]:
  if "fp16" in file_name:
    return 9e-2, 9e-2
  if any(q in file_name for q in ["int8", "uint8", "quantized"]):
    return 4.0, 4.0
  return 4e-3, 3e-2

def validate_repos(models: list[str]):
  download_dir = Path(__file__).parent / "models"
  download_dir.mkdir(parents=True, exist_ok=True)
  for model in models:
    parts = model.split("/")
    if len(parts) == 2:
      repo_id = model
      root_str = download_with_retry(repo_id, ["*.onnx", "*.onnx_data"], download_dir)
      root_path = Path(root_str)
      download_with_retry(repo_id, ["*config.json"], download_dir)
      config = get_config(root_path)
      for onnx_model in root_path.rglob("*.onnx"):
        rtol, atol = get_tolerances(onnx_model.name)
        print(f"Validating {onnx_model.relative_to(root_path)} with rtol={rtol}, atol={atol}")
        run_huggingface_validate(onnx_model, config, rtol, atol)
        print("Passed!")
    else:
      assert parts[-1].endswith(".onnx"), "Last part of path must be an ONNX file"
      repo_id = "/".join(parts[:2])
      rel = "/".join(parts[2:])
      root_str = download_with_retry(repo_id, [rel, "*.onnx_data"], download_dir)
      root_path = Path(root_str)
      download_with_retry(repo_id, ["*config.json"], download_dir)
      config = get_config(root_path)
      rtol, atol = get_tolerances(parts[-1])
      print(f"Validating {model} with rtol={rtol}, atol={atol}")
      run_huggingface_validate(root_path / rel, config, rtol, atol)
      print("Passed!")

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python validate.py <model1> [<model2> ...]")
    sys.exit(1)
  validate_repos(sys.argv[1:])
