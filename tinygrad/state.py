import os, json, pathlib, zipfile, pickle
from typing import Dict, Union
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, prod, argsort
from tinygrad.shape.shapetracker import strides_for_shape

safe_dtypes = {"F16": dtypes.float16, "F32": dtypes.float32, "U8": dtypes.uint8, "I8": dtypes.int8, "I32": dtypes.int32, "I64": dtypes.int64}
inverse_safe_dtypes = {v:k for k,v in safe_dtypes.items()}

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

# torch support!

def torch_load(fn:str):
  t = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}")

  offsets: Dict[str, int] = {}
  lens: Dict[str, int] = {}
  def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    #print(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata)
    lens[storage[2]] = storage[4] * storage[1].itemsize
    if storage[2] not in offsets: return None
    ret = t[offsets[storage[2]]:offsets[storage[2]]+prod(size)].cast(storage[1])

    # 6 lines to deal with permuted tensors. NOTE: this currently requires reading off the disk
    shape_strides = [(s, st) for s,st in zip(size, stride) if s != 1]
    permute_indexes = [len(shape_strides)-1-y for y in argsort([x[1] for x in shape_strides])]
    if tuple(permute_indexes) != tuple(range(len(permute_indexes))):
      intermediate_shape = tuple([shape_strides[x][0] for x in argsort(permute_indexes)])
      assert tuple([shape_strides[i][1] for i in argsort(permute_indexes)]) == strides_for_shape(intermediate_shape), "nonpermutable strides"
      ret = ret.cpu().reshape(intermediate_shape).permute(permute_indexes)

    return ret.reshape(size)

  intercept = {"HalfStorage": dtypes.float16, "FloatStorage": dtypes.float32, "LongStorage": dtypes.int64, "_rebuild_tensor_v2": _rebuild_tensor_v2}
  class TorchPickle(pickle.Unpickler):
    def find_class(self, module, name): return intercept[name] if module.startswith("torch") else super().find_class(module, name)
    def persistent_load(self, pid): return pid

  if tuple(t[0:2].numpy()) == (0x50, 0x4b):
    myzip = zipfile.ZipFile(fn, 'r')
    base_name = myzip.namelist()[0].split('/', 1)[0]
    for n in myzip.namelist():
      if n.startswith(f'{base_name}/data/'):
        with myzip.open(n) as myfile:
          offsets[n.split("/")[-1]] = myfile._orig_compress_start # type: ignore
    with myzip.open(f'{base_name}/data.pkl') as myfile:
      return TorchPickle(myfile).load()
  else:
    with open(fn, "rb") as f:
      pkl = TorchPickle(f)
      _, _, _, rwd, _, ids, base_offset = pkl.load(), pkl.load(), pkl.load(), f.tell(), pkl.load(), pkl.load(), f.tell()
      for i in ids:
        offsets[i] = base_offset + 8
        base_offset += 8 + lens[i]
      f.seek(rwd)
      return TorchPickle(f).load()
