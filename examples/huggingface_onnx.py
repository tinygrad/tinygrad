import argparse, onnx, json, csv
from collections import Counter
from pathlib import Path
from huggingface_hub import list_models, snapshot_download
from tinygrad.helpers import _ensure_downloads_dir
from examples.benchmark_onnx import benchmark

def huggingface_download_onnx_model(model_id:str) -> Path:
  # download all onnx models
  return Path(snapshot_download(repo_id=model_id, allow_patterns=["*.onnx", "*.onnx_data", "*config.json"], cache_dir=_ensure_downloads_dir()))

def get_model_ops(onnx_model:onnx.ModelProto) -> int:
  return dict(Counter(n.op_type for n in onnx_model.graph.node))

def run_huggingface_model(model_id:str):
  report = {"url": f"https://huggingface.co/{model.id}"}
  print(f"Downloading ...")
  root_path = huggingface_download_onnx_model(model_id)
  print(f"Downloaded at {root_path}")

  for onnx_model_path in root_path.rglob("*.onnx"):
    relative_path = str(onnx_model_path.relative_to(root_path))
    report[relative_path] = {}
    print(f"Benchmarking {relative_path}")
    # onnx_model = onnx.load(onnx_model_path)
    # report[relative_path]["ops"] = get_model_ops(onnx_model)

    config_paths = list(root_path.rglob("config.json")) + list(root_path.rglob("preprocessor_config.json"))
    config = {k: v for path in config_paths for k, v in json.load(path.open()).items()}

    try:
      # TODO: pass report into benchmark to collect more stats? Like run speed, inputs chosen, etc
      benchmark(onnx_model_path, config, test_vs_ort=True)
      report[relative_path]["status"] = "success"
    except Exception as e:
      report[relative_path]["status"] = f"failed: {e}"

  return report

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--sort', default="downloads", help="sort by (downloads, download_all_time, trending)", choices=["downloads", "download_all_time", "trending"])
  parser.add_argument('--limit', type=int, default=10, help="number of models") # 100 is alot lol
  parser.add_argument('--model', default="", help="the name of a model.id (repo name) from huggingface to target")
  # parser.add_argument('--onnx-path', default=None, help="path to a specific ONNX model to benchmark. If not provided, benchmarks all ONNX models in the repository.")
  args = parser.parse_args()

  CSV = {}
  if args.model != "":
    print(f"** Running benchmark for {args.model} on huggingface **")
    run_huggingface_model(args.model)
    CSV["url"] = f"https://huggingface.co/{args.model}"
    CSV[args.model] = run_huggingface_model(args.model)
  else:
    print(f"** Running benchmarks for top {args.limit} models ranked by '{args.sort}' on huggingface **")
    for i, model in enumerate(list_models(filter="onnx", sort=args.sort, limit=args.limit)):
      print(f"{i}: {model.id} ({getattr(model, args.sort)} {args.sort}) ")
      CSV[model.id] = run_huggingface_model(model.id)
      # CSV[model.id]["downloads"] = model.downloads
      # CSV[model.id]["download_all_time"] = model.downloads_all_time
      # CSV[model.id]["trending"] = model.trending_score

  print(json.dumps(CSV, indent=2))