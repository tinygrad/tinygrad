import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, prod

import os, json
safe_dtypes = {"F32": dtypes.float32, "U8": dtypes.uint8}
def safe_load(fn):
  t = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}")
  json_len = t[0:1].cast(dtypes.int64).numpy()[0]
  metadata = json.loads(t[8:8+json_len].numpy().tobytes())
  print(metadata)
  return {k:t[8+json_len+v['data_offsets'][0]:].cast(safe_dtypes[v['dtype']])[:prod(v['shape'])].reshape(v['shape']) for k,v in metadata.items()}

def create_demo(fn):
  import torch
  from safetensors.torch import save_file
  tensors = {
    "weight1": torch.randn((16, 16)),
    "weight2": torch.arange(0, 16, dtype=torch.uint8)
  }
  save_file(tensors, fn)
  return tensors

if __name__ == "__main__":
  test = create_demo("/tmp/model.safetensors")
  ret = safe_load("/tmp/model.safetensors")
  print(ret)
  print(ret['weight2'].numpy())
  np.testing.assert_array_equal(ret['weight1'].numpy(), test['weight1'].numpy())
  np.testing.assert_array_equal(ret['weight2'].numpy(), test['weight2'].numpy())
