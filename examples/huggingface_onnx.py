import argparse, onnx, json, csv
from collections import Counter
from pathlib import Path
from huggingface_hub import list_models, snapshot_download
from tinygrad.helpers import _ensure_downloads_dir, fetch
from examples.benchmark_onnx import benchmark

# # I wrote all of this because I didn't know snapshot_download existed...
# # maybe just delete this since it might not be used anymore
# def download(model_id:str):
#   """ downloads the model, external data, config, and preprocessing config from huggingface """
#   base_path = f"https://huggingface.co/{model_id}/resolve/main"
#   potential_file_paths = [
#     "onnx/model.onnx",
#     "model.onnx",
#     "onnx/decoder_model.onnx",
#     "onnx/decoder_model_merged.onnx",
#     "punct_cap_seg_en.onnx", # for "1-800-BAD-CODE/punctuation_fullstop_truecase_english"
#   ]
#   for file_path in potential_file_paths:
#     url = f"{base_path}/{file_path}"
#     model_name = file_path.split('/')[-1]
#
#     # download onnx model
#     try:
#       model_path = fetch(url, model_name, model_id)
#       print(f"Downloaded model at {model_path.as_posix()}")
#     # early continue to the next file model isn't found
#     except urllib.error.HTTPError as e:
#       if e.code == 404: continue
#       raise
#     # raise error if unexpected error occurs
#     except Exception: raise
#
#     # download onnx external data in the same directory
#     try:
#       file_path_no_extension = url.rsplit('.', 1)[0]
#       external_data_url = f"{file_path_no_extension}.onnx_data"
#       external_data_name = external_data_url.split('/')[-1]
#       external_data_path = fetch(external_data_url, external_data_name, model_id)
#       print(f"Downloaded external data at {external_data_path.as_posix()}")
#     except urllib.error.HTTPError as e:
#       if e.code != 404: raise
#     except Exception: raise
#
#     # download configs
#     for config_path in (base_path, base_path + "/onnx"):
#       try:
#         preprocessor_config = fetch(f"{config_path}/preprocessor_config.json", "preprocessor_config.json", model_id)
#         print(f"Downloaded preprocessor config at {preprocessor_config.as_posix()}")
#         break
#       except urllib.error.HTTPError as e:
#         if e.code != 404: raise
#       except Exception: raise
#     for config_path in (base_path, base_path + "/onnx"):
#       try:
#         model_config = fetch(f"{config_path}/config.json", "config.json", model_id)
#         print(f"Downloaded model config at {model_config.as_posix()}")
#         break
#       except urllib.error.HTTPError as e:
#         if e.code != 404: raise
#       except Exception: raise
#
#     # yield the model path
#     yield model_path
#
#   raise Exception(f"failed to download model from https://huggingface.co/{model_id}")

def huggingface_download_onnx_model(model_id:str) -> Path:
  # download all onnx models
  return Path(snapshot_download(repo_id=model_id, allow_patterns=["*.onnx", "*.onnx_data", "*config.json"], cache_dir=_ensure_downloads_dir()))

def get_model_ops(onnx_model:onnx.ModelProto) -> int:
  return Counter(n.op_type for n in onnx_model.graph.node)

def run_huggingface_model(model_id:str):
  print(f"Downloading ...")
  root_path = huggingface_download_onnx_model(model_id)
  print(f"Downloaded at {root_path}")

  for onnx_model_path in root_path.rglob("*.onnx"):
    relative_path = onnx_model_path.relative_to(root_path)
    print(f"Benchmarking {relative_path}")
    onnx_model = onnx.load(onnx_model_path)
    print(f"Op distribution: {get_model_ops(onnx_model)}")

    config_paths = list(root_path.rglob("config.json")) + list(root_path.rglob("preprocessor_config.json"))
    config = {k: v for path in config_paths for k, v in json.load(path.open()).items()}

    try:
      benchmark(onnx_model_path, config, test_vs_ort=True)
    except Exception as e:
      print(f"Error: {e}")
      print(f"Failed to benchmark {relative_path}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--sort', default="downloads", help="sort by (downloads, download_all_time, trending)", choices=["downloads", "download_all_time", "trending"])
  parser.add_argument('--limit', type=int, default=10, help="number of models") # 100 is alot lol
  parser.add_argument('--model', default="", help="the name of a model.id (repo name) from huggingface to target")
  # parser.add_argument('--onnx-path', default=None, help="path to a specific ONNX model to benchmark. If not provided, benchmarks all ONNX models in the repository.")
  args = parser.parse_args()

  if args.model != "":
    print(f"** Running benchmark for {args.model} on huggingface **")
    run_huggingface_model(args.model)
  else:
    print(f"** Running benchmarks for top {args.limit} models ranked by '{args.sort}' on huggingface **")
    for i, model in enumerate(list_models(filter="onnx", sort=args.sort, limit=args.limit)):
      print(f"{i}: {model.id} ({getattr(model, args.sort)} {args.sort}) ")
      run_huggingface_model(model.id)