import onnx, json, sys
from pathlib import Path
from tinygrad.frontend.onnx import OnnxRunner
from extra.onnx_helpers import validate, get_example_inputs
from huggingface_hub import snapshot_download

def get_config(root_path: Path):
  ret = {}
  for path in root_path.rglob("*config.json"):
    config = json.load(path.open())
    if isinstance(config, dict):
      ret.update(config)
  return ret

def run_huggingface_validate(onnx_model_path, config, rtol, atol):
  onnx_model = onnx.load(onnx_model_path)
  onnx_runner = OnnxRunner(onnx_model)
  inputs = get_example_inputs(onnx_runner.graph_inputs, config)
  validate(onnx_model_path, inputs, rtol=rtol, atol=atol)

def get_tolerances(file_name): # -> rtol, atol
  # TODO very high rtol atol
  if "fp16" in file_name: return 9e-2, 9e-2
  if any(q in file_name for q in ["int8", "uint8", "quantized"]): return 4, 4
  return 4e-3, 3e-2

def debug_run(model_path, config, rtol, atol):
  run_huggingface_validate(model_path, config, rtol, atol)

def validate_repos(models:list[str]):
  download_dir = Path(__file__).parent / "models"
  for model in models:
    path = model.split("/")
    if len(path) == 2:
      # repo id
      # validates all onnx models inside repo
      repo_id = "/".join(path)
      root_path = Path(snapshot_download(repo_id=repo_id, allow_patterns=["*.onnx", "*.onnx_data"], cache_dir=download_dir))
      snapshot_download(repo_id=repo_id, allow_patterns=["*config.json"], cache_dir=download_dir)
      config = get_config(root_path)
      for onnx_model in root_path.rglob("*.onnx"):
        rtol, atol = get_tolerances(onnx_model.name)
        print(f"validating {onnx_model.relative_to(root_path)} with {rtol=}, {atol=}")
        run_huggingface_validate(onnx_model, config, rtol, atol)
    else:
      # model id
      # only validate the specified onnx model
      onnx_model = path[-1]
      assert path[-1].endswith(".onnx")
      repo_id, relative_path = "/".join(path[:2]), "/".join(path[2:])
      print(relative_path)
      root_path = Path(snapshot_download(repo_id=repo_id, allow_patterns=[str(relative_path), "*.onnx_data"], cache_dir=download_dir))
      snapshot_download(repo_id=repo_id, allow_patterns=["*config.json"], cache_dir=download_dir)
      config = get_config(root_path)
      rtol, atol = get_tolerances(onnx_model)
      print(f"validating {model} with {rtol=}, {atol=}")
      run_huggingface_validate(root_path / relative_path, config, rtol, atol)
      print("passed!")

if __name__ == "__main__":
  models = sys.argv[1:]
  validate_repos(models)
