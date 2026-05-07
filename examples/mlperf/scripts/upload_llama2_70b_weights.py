import hashlib
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import CommitOperationDelete, HfApi
from tqdm import tqdm

from tinygrad import Tensor
from tinygrad.helpers import DEV
from tinygrad.nn.state import safe_load, safe_save
from extra.models.llama import convert_from_huggingface, precompute_freqs_cis
from extra.huggingface_onnx.huggingface_manager import DOWNLOADS_DIR, snapshot_download_with_retry
from examples.mlperf.model_train import LLAMA2_70B_ARGS

DEV.value = 'CPU'

HF_REPO_ID = "imaolo/llama2-70b-fused-qkv-flat-mlperf"
HF_REF_REPO_ID = "regisss/llama2-70b-fused-qkv-mlperf"
HF_REF_REVISION = "647cb0c8858ddefd10231a20ddfa68e4eb5e850e"
HF_REF_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.py"]
HF_REF_EXPECTED_HASH = "986e507271274d2a6563974787af9ac7"

REF_WEIGHTS_PATH = DOWNLOADS_DIR/HF_REF_REPO_ID
WEIGHTS_PATH = DOWNLOADS_DIR/HF_REPO_ID

def hash_file_md5(file_path:Path, chunk_size:int=4194304) -> str|None:
  md5_hash = hashlib.md5()
  try:
    with open(file_path, "rb") as f:
      while chunk := f.read(chunk_size):
        md5_hash.update(chunk)
  except FileNotFoundError:
    return None
  return md5_hash.hexdigest()

def hash_directory(path:Path) -> str:
  hashes = []
  with ThreadPoolExecutor() as executor:
    futures = [executor.submit(hash_file_md5, file_path) for file_path in path.rglob("*") if file_path.is_file()]
    for future in tqdm(futures, desc="hash reference weights", unit="file"):
      if file_hash := future.result(): hashes.append(file_hash)

  combined_hash = hashlib.md5()
  for file_hash in sorted(hashes):
    combined_hash.update(file_hash.encode())
  return combined_hash.hexdigest()

def remove_hidden_dirs(path:Path) -> None:
  for dir_path in sorted((p for p in path.rglob(".*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
    shutil.rmtree(dir_path)

def verify_reference_weights() -> None:
  directory_hash = hash_directory(REF_WEIGHTS_PATH)
  assert directory_hash == HF_REF_EXPECTED_HASH, f"expected reference weight hash {HF_REF_EXPECTED_HASH}, got {directory_hash}"
  print(f"verified reference weights with hash {directory_hash}")

def download_reference_weights() -> None:
  print(f"downloading reference weights to {REF_WEIGHTS_PATH}")
  REF_WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
  snapshot_download_with_retry(repo_id=HF_REF_REPO_ID, revision=HF_REF_REVISION, local_dir=REF_WEIGHTS_PATH, allow_patterns=HF_REF_ALLOW_PATTERNS)
  remove_hidden_dirs(REF_WEIGHTS_PATH)
  verify_reference_weights()
  print("downloaded reference weights")

def load_reference_state_dict() -> dict[str, Tensor]:
  ref_weight_paths = sorted(REF_WEIGHTS_PATH.glob("*.safetensors"))
  assert len(ref_weight_paths) == 29, f"expected 29 weight files, found {len(ref_weight_paths)}"
  ref_state_dict = {}
  for weight_file in tqdm(ref_weight_paths, desc="load reference shards", unit="file"):
    ref_state_dict.update(safe_load(weight_file))
  return convert_from_huggingface(ref_state_dict, LLAMA2_70B_ARGS["n_layers"], LLAMA2_70B_ARGS["n_heads"], LLAMA2_70B_ARGS["n_kv_heads"])

def pop_stacked_layers(state_dict:dict[str, Tensor], key_fmt:str) -> Tensor:
  return Tensor.stack([state_dict.pop(key_fmt.format(i=i)) for i in range(LLAMA2_70B_ARGS["n_layers"])])

def build_flat_state_dict(ref_state_dict:dict[str, Tensor]) -> dict[str, Tensor]:
  flat_state_dict: dict[str, Tensor] = {}
  flat_state_dict["attention_norm"] = pop_stacked_layers(ref_state_dict, "layers.{i}.attention_norm.weight")
  flat_state_dict["ffn_norm"] = pop_stacked_layers(ref_state_dict, "layers.{i}.ffn_norm.weight")
  flat_state_dict["norm.weight"] = ref_state_dict.pop("norm.weight")
  flat_state_dict["output"] = ref_state_dict.pop("output.weight").unsqueeze(dim=0)
  flat_state_dict["tok_embeddings.weight"] = ref_state_dict.pop("tok_embeddings.weight")
  flat_state_dict["w1"] = pop_stacked_layers(ref_state_dict, "layers.{i}.feed_forward.w1.weight")
  flat_state_dict["w2"] = pop_stacked_layers(ref_state_dict, "layers.{i}.feed_forward.w2.weight")
  flat_state_dict["w3"] = pop_stacked_layers(ref_state_dict, "layers.{i}.feed_forward.w3.weight")
  flat_state_dict["wo"] = pop_stacked_layers(ref_state_dict, "layers.{i}.attention.wo.weight")
  flat_state_dict["wqkv"] = pop_stacked_layers(ref_state_dict, "layers.{i}.attention.wqkv.weight")
  flat_state_dict["w13"] = flat_state_dict.pop("w1").to('CPU').cat(flat_state_dict.pop("w3").to('CPU'), dim=1).realize()
  return flat_state_dict

def clear_dir(path: Path, pattern:str="*.safetensors") -> None:
  path.mkdir(parents=True, exist_ok=True)
  for tensor_path in tqdm((p for p in path.glob(pattern) if p.is_file()), desc="clear local tensors", unit="file"):
    tensor_path.unlink()

def save_state_dict(state_dict:dict[str, Tensor]) -> list[Path]:
  weight_files = [WEIGHTS_PATH / f"{name}.safetensors" for name in state_dict.keys()]
  for file_name, (name, tensor) in tqdm(zip(weight_files, state_dict.items()), total=len(weight_files), desc="saving flat weight shards"):
    safe_save({name: tensor}, file_name)
  return weight_files

def upload_files(files:list[Path]):
  api = HfApi()
  api.create_repo(repo_id=HF_REPO_ID, exist_ok=True)
  remote_files = api.list_repo_files(repo_id=HF_REPO_ID)
  if remote_files:
    api.create_commit(
      repo_id=HF_REPO_ID,
      operations=[(CommitOperationDelete(path_in_repo=path)) for path in remote_files],
      commit_message="Delete existing repo contents before uploading rebuilt flat weights",
    )
  return api.upload_folder(
    folder_path=WEIGHTS_PATH,
    repo_id=HF_REPO_ID,
    allow_patterns=[p.name for p in files],
    commit_message=f"Uploaded {len(files)} rebuilt flat weights",
  )

def main() -> None:
  # cleanup
  clear_dir(REF_WEIGHTS_PATH, "**/*")
  clear_dir(WEIGHTS_PATH)

  # download
  download_reference_weights()

  # load reference weights
  ref_state_dict = load_reference_state_dict()
  for t in tqdm(ref_state_dict.values(), desc="realizing ref weights: "):
    t.to_('CPU').realize()

  flat_state_dict = build_flat_state_dict(ref_state_dict)
  del ref_state_dict

  # realize new tensors
  for t in tqdm(flat_state_dict.values(), desc="realizing flat weights: "):
    t.to_('CPU').realize()

  # save and upload
  weight_files = save_state_dict(flat_state_dict)
  print(upload_files(weight_files))

if __name__ == "__main__":
  main()
