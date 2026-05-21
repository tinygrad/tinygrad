from pathlib import Path

from huggingface_hub import CommitOperationDelete, HfApi
from tqdm import tqdm

from tinygrad import Tensor
from tinygrad.helpers import DEV
from tinygrad.nn.state import safe_load, safe_save
from extra.models.llama import convert_from_huggingface
from extra.huggingface_onnx.huggingface_manager import DOWNLOADS_DIR, snapshot_download_with_retry
from examples.mlperf.model_train import LLAMA2_70B_ARGS
from examples.mlperf.helpers import clean_dir
from examples.mlperf.hash import hash_directory

## 1. download reference weights
## 2. reshape them for flat_llama
## 3. upload reshaped weights

DEV.value = 'CPU'

HF_REPO_ID = "imaolo/llama2-70b-fused-qkv-flat-mlperf"
HF_REF_REPO_ID = "regisss/llama2-70b-fused-qkv-mlperf"
HF_REF_REVISION = "647cb0c8858ddefd10231a20ddfa68e4eb5e850e"
HF_REF_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.py"]
HF_REF_EXPECTED_HASH = "986e507271274d2a6563974787af9ac7"
HF_EXPECTED_HASH = "9931046352d60e1687bef38094899e99"

REF_WEIGHTS_PATH = DOWNLOADS_DIR/HF_REF_REPO_ID
WEIGHTS_PATH = DOWNLOADS_DIR/HF_REPO_ID

def download_reference_weights() -> None:
  print(f"downloading reference weights to {REF_WEIGHTS_PATH}")
  snapshot_download_with_retry(repo_id=HF_REF_REPO_ID, revision=HF_REF_REVISION, local_dir=REF_WEIGHTS_PATH, allow_patterns=HF_REF_ALLOW_PATTERNS)
  clean_dir(REF_WEIGHTS_PATH, [patt[1:] for patt in HF_REF_ALLOW_PATTERNS])
  assert hash_directory(REF_WEIGHTS_PATH) == HF_REF_EXPECTED_HASH
  print("downloaded & verified reference weights")

def load_reference_state_dict() -> dict[str, Tensor]:
  ref_weight_paths = sorted(REF_WEIGHTS_PATH.glob("*.safetensors"))
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

def save_state_dict(state_dict:dict[str, Tensor]) -> list[Path]:
  weight_files = [WEIGHTS_PATH / f"{name}.safetensors" for name in state_dict.keys()]
  for file_name, (name, tensor) in tqdm(zip(weight_files, state_dict.items()), total=len(weight_files), desc="saving flat weights"):
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

if __name__ == "__main__":
  import shutil
  # cleanup
  shutil.rmtree(REF_WEIGHTS_PATH, ignore_errors=True)
  shutil.rmtree(WEIGHTS_PATH, ignore_errors=True)
  REF_WEIGHTS_PATH.mkdir(parents=True)
  WEIGHTS_PATH.mkdir(parents=True)

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

  # verify hash
  assert hash_directory(WEIGHTS_PATH) == HF_EXPECTED_HASH

  print(upload_files(weight_files))
