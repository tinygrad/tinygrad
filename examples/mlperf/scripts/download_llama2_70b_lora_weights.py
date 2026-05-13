from pathlib import Path
from tinygrad.helpers import getenv
from examples.mlperf.helpers import clean_dir
from examples.mlperf.hash import hash_directory
from extra.huggingface_onnx.huggingface_manager import snapshot_download_with_retry

MODEL_PATH = getenv("MODEL_PATH", "/raid/weights/c4-llama2-70b-lora/")
REPO_ID = "imaolo/llama2-70b-fused-qkv-flat-mlperf"
EXPECTED_HASH = "9931046352d60e1687bef38094899e99"
REVISION = "5ef07c7bdfffa493f16d7ffa394a203fc5ee8734"

if __name__ == '__main__':
  weights_path = Path(snapshot_download_with_retry(repo_id=REPO_ID, local_dir=MODEL_PATH, revision=REVISION, allow_patterns=["*safetensors*", "*.json", "*.md"]))

  clean_dir(weights_path, ['.safetensors'])

  assert hash_directory(weights_path) == EXPECTED_HASH