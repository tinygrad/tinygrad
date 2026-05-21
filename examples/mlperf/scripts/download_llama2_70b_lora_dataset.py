import numpy as np
from pathlib import Path
from tinygrad.helpers import getenv
from examples.mlperf.helpers import clean_dir
from examples.mlperf.dataloader import _load_llama2_70b_lora_split
from examples.mlperf.hash import hash_directory
from huggingface_hub import snapshot_download

REPO = "regisss/scrolls_gov_report_preprocessed_mlperf_2"
REVISION = "21ff1233ee3e87bc780ab719c755170148aba1cb"
HASH = "682a5f40b790a56751bf8303554efc08"

def download_dataset(base_dir: Path) -> Path:
  base_dir = Path(snapshot_download(repo_id=REPO, revision=REVISION, repo_type="dataset", local_dir=base_dir, allow_patterns="*.parquet"))
  clean_dir(data_dir:=(base_dir / 'data'), [".parquet"])

  # AMD: https://github.com/mlcommons/training_results_v5.1/blob/main/AMD/benchmarks/llama2_70b_lora/implementations/MI350X_EPYC_9575F_pytorch_llama2_70b/scripts/download_dataset.py#L25-L41
  # Cisco: https://github.com/mlcommons/training_results_v5.1/blob/main/Cisco/benchmarks/llama2_70b_lora/implementations/pytorch/scripts/download_dataset.py#L25-L41
  # QCT: https://github.com/mlcommons/training_results_v5.1/blob/main/Quanta_Cloud_Technology/benchmark/llama2_70b_lora/implementations/pytorch_D74H-7U/scripts/download_dataset.py#L25-L41
  assert (digest:=hash_directory(data_dir)) == HASH, f"llama2 70b lora dataset hash {digest} != {HASH}"

  for split in ['train', 'validation']: verify_dataset_split(split, *_load_llama2_70b_lora_split(base_dir, split))
  return base_dir

def verify_dataset_split(split: str, input_ids: np.ndarray, labels: np.ndarray):
  assert input_ids.shape == labels.shape, f"{split} input_ids shape {input_ids.shape} != labels shape {labels.shape}"
  assert not (input_ids == -100).any(), f"{split} input_ids contains -100"
  if split == "train": assert np.array_equal(input_ids, labels), "train labels differ from input_ids"
  else: assert (labels == -100).any(), "validation labels contain no -100 mask"

if __name__ == '__main__':
  download_dataset(getenv("BASEDIR", "/raid/datasets/c4-llama2-70b-lora/"))