import os, json, pathlib
from typing import Dict, Union
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, prod

safe_dtypes = {"F16": dtypes.float16, "F32": dtypes.float32, "U8": dtypes.uint8, "I8": dtypes.int8, "I32": dtypes.int32, "I64": dtypes.int64}
inverse_safe_dtypes = {v:k for k,v in safe_dtypes.items()}

def torch_load(fn:str):
  import zipfile, pickle
  myzip = zipfile.ZipFile(fn, 'r')
  base_name = myzip.namelist()[0].split('/', 1)[0]
  t = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}")

  def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    #print(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata)
    with myzip.open(f'{base_name}/data/{storage[2]}') as myfile:
      offset = myfile._orig_compress_start  # type: ignore
    return t[offset:offset+prod(size)].cast(storage[1]).reshape(size)

  intercept = {"HalfStorage": dtypes.float16, "_rebuild_tensor_v2": _rebuild_tensor_v2}
  class TorchPickle(pickle.Unpickler):
    def find_class(self, module, name):
      if module.startswith("torch"): return intercept[name]
      return super().find_class(module, name)
    def persistent_load(self, pid): return pid

  with myzip.open(f'{base_name}/data.pkl') as myfile: return TorchPickle(myfile).load()

def safe_load(fn:Union[Tensor,str]) -> Dict[str, Tensor]:
  t = fn if isinstance(fn, Tensor) else Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}")
  json_len = t[0:1].cast(dtypes.int64).numpy()[0]
  metadata = json.loads(t[8:8+json_len].numpy().tobytes())
  return {k:t[8+json_len+v['data_offsets'][0]:].cast(safe_dtypes[v['dtype']])[:prod(v['shape'])].reshape(v['shape']) for k,v in metadata.items() if k != "__metadata__"}

def safe_save(tensors:Dict[str, Tensor], fn:str):
  metadata, offset = {}, 0
  for k,v in tensors.items():
    metadata[k] = {'dtype': inverse_safe_dtypes[v.dtype], 'shape': list(v.shape), 'data_offsets':[offset, offset+v.nbytes()]}
    offset += v.nbytes()
  j = json.dumps(metadata, separators=(',', ':'))
  j += "\x20"*((8-len(j)%8)%8)
  pathlib.Path(fn).unlink(missing_ok=True)
  t = Tensor.empty(8+len(j)+offset, dtype=dtypes.uint8, device=f"disk:{fn}")
  t[0:1].cast(dtypes.int64).assign([len(j)])
  t[8:8+len(j)].assign(Tensor(list(j.encode('utf-8')), dtype=dtypes.uint8))
  for k,v in safe_load(t).items(): v.assign(tensors[k])

# TODO: move get_state_dict and get_parameters here
from tinygrad.nn.optim import get_state_dict, get_parameters  # pylint: disable=unused-import # noqa: F401
