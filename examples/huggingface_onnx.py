import argparse, onnx, json
from collections import Counter
from pathlib import Path
from huggingface_hub import list_models, snapshot_download
from tinygrad import Tensor
from tinygrad.tensor import _to_np_dtype
from tinygrad.helpers import _ensure_downloads_dir
from extra.onnx import OnnxValue, OnnxRunner
import onnxruntime as ort
import numpy as np

def get_example_inputs(graph_inputs:dict[str, OnnxValue]):
  ret: dict[str, Tensor] = {}
  for name, spec in graph_inputs.items():
    assert not spec.is_optional and not spec.is_sequence, "only allow tensor input for now"
    shape = tuple(dim if isinstance(dim, int) else 1 for dim in spec.shape)
    value = Tensor(np.random.uniform(size=shape).astype(_to_np_dtype(spec.dtype)) * 8).realize()
    ret.update({name:value})
  return ret

def validate(fp, config=None, rtol=1e-5, atol=1e-5):
  run_onnx = OnnxRunner(onnx.load(fp))
  new_inputs = get_example_inputs(run_onnx.graph_inputs)
  tinygrad_out = run_onnx(new_inputs)

  ort_options = ort.SessionOptions()
  ort_options.log_severity_level = 3
  ort_sess = ort.InferenceSession(fp, ort_options, ["CPUExecutionProvider"])
  np_inputs = {k:v.numpy() for k,v in new_inputs.items()}
  out_names = list(run_onnx.graph_outputs)
  out_values = ort_sess.run(out_names, np_inputs)
  ort_out = dict(zip(out_names, out_values))

  assert len(tinygrad_out) == len(ort_out) and tinygrad_out.keys() == ort_out.keys()
  for k in tinygrad_out.keys():
    tiny_v, onnx_v = tinygrad_out[k], ort_out[k]
    if tiny_v is None: assert tiny_v == onnx_v
    else: np.testing.assert_allclose(tiny_v.numpy(), onnx_v, rtol=rtol, atol=atol, err_msg=f"For tensor '{k}' in {tinygrad_out.keys()}")

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
      validate(onnx_model_path, config)
      report[relative_path]["status"] = "success"
    except Exception as e:
      report[relative_path]["status"] = f"failed: {e}"

  return report

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--sort', default="downloads", help="sort by (downloads, download_all_time, trending)", choices=["downloads", "download_all_time", "trending"])
  parser.add_argument('--limit', type=int, default=10, help="number of models") # 100 is alot lol
  parser.add_argument('--repo', default="", help="the name of a model.id (repo name) from huggingface to target")
  parser.add_argument('--model', default=None, help="path to a specific ONNX model to benchmark. If not provided, benchmarks all ONNX models in the repository.")
  args = parser.parse_args()

  d = {}
  if args.repo != "":
    print(f"** Running benchmark for {args.repo}/{args.model or ''} on huggingface **")
    d["url"] = f"https://huggingface.co/{args.repo}"
    d[args.repo] = run_huggingface_model(args.repo, args.model)
  else:
    print(f"** Running benchmarks for top {args.limit} models ranked by '{args.sort}' on huggingface **")
    for i, model in enumerate(list_models(filter="onnx", sort=args.sort, limit=args.limit)):
      print(f"{i}: {model.id} ({getattr(model, args.sort)} {args.sort}) ")
      d[model.id] = run_huggingface_model(model.id)
      # d[model.id]["downloads"] = model.downloads
      # d[model.id]["download_all_time"] = model.downloads_all_time
      # d[model.id]["trending"] = model.trending_score

  print(json.dumps(d, indent=2))