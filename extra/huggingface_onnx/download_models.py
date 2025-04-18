import yaml, argparse, collections, onnx
from pathlib import Path
from huggingface_hub import snapshot_download
from tinygrad.frontend.onnx import OnnxRunner
from extra.onnx import get_onnx_ops

def download_models(metadata: dict, download_dir: str, sel: int) -> dict:
  """ downloads the models and updates the metadata """
  n = len(metadata["repositories"])
  model_ops = collections.defaultdict(collections.Counter)
  supported_ops = get_onnx_ops()

  for i, (model_id, model_data) in enumerate(metadata["repositories"].items()):
    print(f"Downloading {i+1}/{n}: {model_id}...")
    allow_patterns = [file_info["file"] for file_info in model_data["files"]]
    root_path = Path(snapshot_download(repo_id=model_id, allow_patterns=allow_patterns, cache_dir=download_dir))
    # download configs too (the sizes are small)
    snapshot_download(repo_id=model_id, allow_patterns=["*config.json"], cache_dir=download_dir)
    print(f"Downloaded model files to: {root_path}")

    for onnx_file in allow_patterns:
      if not onnx_file.endswith(".onnx"): continue
      onnx_model = root_path / onnx_file
      try:
        onnx_runner = OnnxRunner(onnx.load(onnx_model))
        for node in onnx_runner.graph_nodes: model_ops[model_id + "/" + onnx_file][node.op] += 1
      except NotImplementedError:
        pass

  model_op_sets = {model: set(ops.keys()) for model, ops in model_ops.items()}
  first_model = next(iter(model_ops.keys()))
  seen_ops = {*list(model_ops[first_model])}

  diverse_models = [first_model]
  for _ in range(sel):
    nm,ops = max(model_op_sets.items(), key=lambda item: len(item[1].difference(seen_ops)))
    if ops.issubset(seen_ops): break
    diverse_models.append(nm)
    seen_ops.update(ops)

  op_counter = sum(model_ops.values(), collections.Counter())
  metadata["stats"].update({
    "model_ops": {key: dict(value.most_common()) for key, value in model_ops.items()},
    "total_op_counter": dict(op_counter.most_common()),
    "unsupported_ops": list(set(op_counter).difference(set(supported_ops))),
    "diverse_models": diverse_models
  })

  return metadata

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Download models from Huggingface Hub based on a YAML configuration file.")
  parser.add_argument("input", type=str, help="Path to the input YAML configuration file containing model information.")
  parser.add_argument("--diversity", type=int, default=10, help="Number of diverse_models to select")
  args = parser.parse_args()

  models_folder = Path(__file__).parent / "models"
  models_folder.mkdir(parents=True, exist_ok=True)
  with open(args.input, 'r') as f:
    metadata = yaml.safe_load(f)
  metadata = download_models(metadata, str(models_folder), args.diversity)

  # Save the updated metadata back to the YAML file
  with open(args.input, 'w') as f: yaml.dump(metadata, f, sort_keys=False)
  print("Download completed according to YAML file.")