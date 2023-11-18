import json, math
import numpy as np
from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor, dtypes
from extra.models.bert import Bert
from extra.dist import dist

if __name__ == "__main__":
  if getenv("DIST"):
    dist.preinit()

if getenv('HALF', 0):
  Tensor.default_type = dtypes.float16
  np_dtype = np.float16
else:
  Tensor.default_type = dtypes.float32
  np_dtype = np.float32

BS, EVAL_BS, STEPS, WARMUP_STEPS, EPOCH,  = getenv("BS", 32), getenv('EVAL_BS', 8), getenv("STEPS", 100000), getenv("WARMUP_STEPS", 10000), getenv("EPOCHS", 30)
EVAL_FREQ = math.floor(0.05*(230.23 * BS + 3000000), 25000)

def get_model_config(path:str):
  with open(path, 'r') as f:
    config = json.load(f)
  return config

def pretrain():
  ...