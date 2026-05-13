from pathlib import Path
from tinygrad.helpers import getenv
from examples.mlperf.helpers import clean_dir
from examples.mlperf.hash import hash_directory
from extra.huggingface_onnx.huggingface_manager import snapshot_download_with_retry

MODEL_PATH = getenv("MODEL_PATH", "/raid/weights/c4-llama2-70b-lora/")
LLAMA2_70B_REPO_ID = "imaolo/llama2-70b-fused-qkv-flat-mlperf"
HF_FLAT_EXPECTED_HASH = ""
HF_FLAT_REVISION = ""

if __name__ == '__main__':
  weights_path = Path(snapshot_download_with_retry(repo_id=LLAMA2_70B_REPO_ID, local_dir=MODEL_PATH, allow_patterns=["*safetensors*", "*.json", "*.md"]))

  clean_dir(weights_path, ['.safetensors'])

  assert hash_directory(weights_path) == HF_FLAT_EXPECTED_HASH