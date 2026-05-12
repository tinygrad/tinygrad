from tinygrad.helpers import getenv
from examples.mlperf.dataloader import download_llama2_70b_lora_dataset
if __name__ == '__main__':
  download_llama2_70b_lora_dataset(getenv("BASEDIR", "/raid/datasets/c4-70b-lora/"))