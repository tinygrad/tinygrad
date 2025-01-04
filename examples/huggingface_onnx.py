import argparse, onnx, json
from collections import Counter
from pathlib import Path
from huggingface_hub import list_models, snapshot_download
from tinygrad.helpers import _ensure_downloads_dir, getenv
from examples.benchmark_onnx import benchmark

def huggingface_download_onnx_model(model_id:str) -> Path:
  # download all onnx models
  return Path(snapshot_download(repo_id=model_id, allow_patterns=["*.onnx", "*.onnx_data", "*config.json"], cache_dir=_ensure_downloads_dir()))

def get_model_ops(onnx_model:onnx.ModelProto) -> dict:
  return dict(Counter(n.op_type for n in onnx_model.graph.node))

def run_huggingface_model(model_id:str, model_path:str|None=None) -> dict:
  report = {"url": f"https://huggingface.co/{model_id}"}
  print("Downloading ...")
  root_path = huggingface_download_onnx_model(model_id)
  print(f"Downloaded at {root_path}")

  onnx_model_paths = root_path.rglob("*.onnx") if model_path is None else [root_path / model_path]
  for onnx_model_path in onnx_model_paths:
    relative_path = str(onnx_model_path.relative_to(root_path))
    report[relative_path] = {}
    print(f"Benchmarking {relative_path}")

    config_paths = list(root_path.rglob("config.json")) + list(root_path.rglob("preprocessor_config.json"))
    config = {k: v for path in config_paths for k, v in json.load(path.open()).items()}

    try:
      # TODO: pass report into benchmark to collect more stats? Like run speed, inputs chosen, etc
      benchmark(onnx_model_path, config, test_vs_ort=int(getenv("ORT", "1")))
      report[relative_path]["status"] = "success"
    except Exception as e:
      report[relative_path]["status"] = f"failed: {e}"

  return report

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--sort', default="downloads", help="sort by (downloads, download_all_time, trending)", choices=["downloads", "download_all_time", "trending"])
  parser.add_argument('--limit', type=int, default=10, help="number of models") # 100 is alot lol
  parser.add_argument('--model', default="", help="the name of a model.id (repo name) from huggingface to target")
  parser.add_argument('--onnx-path', default=None, help="path to a specific ONNX model to benchmark. If not provided, benchmarks all ONNX models in the repository.")
  args = parser.parse_args()

  d = {}
  if args.model != "":
    print(f"** Running benchmark for {args.model}/{args.onnx_path or ''} on huggingface **")
    d["url"] = f"https://huggingface.co/{args.model}"
    d[args.model] = run_huggingface_model(args.model, args.onnx_path)
  else:
    print(f"** Running benchmarks for top {args.limit} models ranked by '{args.sort}' on huggingface **")
    for i, model in enumerate(list_models(filter="onnx", sort=args.sort, limit=args.limit)):
      print(f"{i}: {model.id} ({getattr(model, args.sort)} {args.sort}) ")
      d[model.id] = run_huggingface_model(model.id)
      # d[model.id]["downloads"] = model.downloads
      # d[model.id]["download_all_time"] = model.downloads_all_time
      # d[model.id]["trending"] = model.trending_score

  print(json.dumps(d, indent=2))