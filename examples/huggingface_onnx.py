import onnx, json, os
from pathlib import Path
from huggingface_hub import list_models, snapshot_download
from tinygrad.helpers import _ensure_downloads_dir, getenv
from extra.onnx import OnnxRunner
from extra.onnx_helpers import validate, get_example_inputs

HUGGINGFACE_URL = "https://huggingface.co"
SKIPPED_FILES = ["limit", "avx2", "arm64", "avx512", "avx512_vnni"]
SKIPPED_REPOS = ["stabilityai/stable-diffusion-xl-base-1.0"]

def huggingface_download_onnx_model(model_id:str):
  return Path(snapshot_download(repo_id=model_id, allow_patterns=["*.onnx", "*.onnx_data", "*config.json"], cache_dir=_ensure_downloads_dir()))

def get_config(root_path:Path):
  config_paths = list(root_path.rglob("config.json")) + list(root_path.rglob("preprocessor_config.json"))
  return {k:v for path in config_paths for k,v in json.load(path.open()).items()}

def run_huggingface_benchmark(onnx_model_path, config):
  inputs = get_example_inputs(OnnxRunner(onnx.load(onnx_model_path)).graph_inputs, config)
  validate(onnx_model_path, inputs, getenv("ONNXLIMIT", -1), rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
  assert getenv("LIMIT") or getenv("MODELPATH", ""), "ex: LIMIT=25 or MODELPATH=google-bert/bert-base-uncased/model.onnx"

  # for running
  if limit := getenv("LIMIT"):
    sort = "downloads"
    result = {"passed": 0, "failed": 0}
    print(f"** Running benchmarks on top {limit} models ranked by '{sort}' on huggingface **")
    for i, model in enumerate(list_models(filter="onnx", sort=sort, limit=limit)):
      if model.id in SKIPPED_REPOS: continue  # skip these
      print(f"{i}: ({getattr(model, sort)} {sort}) ")
      url = f"{HUGGINGFACE_URL}/{model.id}"
      result[model.id] = {"url": url}
      print(f"Downloading: {url}")
      root_path = huggingface_download_onnx_model(model.id)
      print(f"Saved: {root_path}")
      for onnx_model_path in root_path.rglob("*.onnx"):
        if any(skip in onnx_model_path.stem for skip in SKIPPED_FILES): continue  # skip these
        relative_path = str(onnx_model_path.relative_to(root_path))
        print(f"Benchmarking {relative_path}")
        try:
          run_huggingface_benchmark(onnx_model_path, get_config(root_path))
          result[model.id][relative_path] = {"status": "passed"}
          result["passed"] += 1
        except Exception as e:
          result[model.id][relative_path] = {"status": f"failed {e}"}
          result["failed"] += 1

    with open("huggingface_results.json", "w") as f:
      json.dump(result, f, indent=2)
      print(f"report saved to {os.path.abspath('huggingface_results.json')}")

  # for debug
  if model_path := str(getenv("MODELPATH", "")):
    model_id, relative_path = model_path.split("/", 2)[:2], model_path.split("/", 2)[2]
    model_id = "/".join(model_id)
    root_path = huggingface_download_onnx_model(model_id)
    run_huggingface_benchmark(root_path / relative_path, get_config(root_path))
