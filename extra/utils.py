import pickle
import numpy as np
from tqdm import tqdm
import tempfile, platform
from collections import defaultdict
from tinygrad.helpers import prod, getenv, DEBUG, dtypes
from tinygrad.ops import GlobalCounters
from tinygrad.tensor import Tensor
from tinygrad.lazy import Device
from tinygrad.shape.shapetracker import strides_for_shape
OSX = platform.system() == "Darwin"

def fetch(url):
  if url.startswith("/"):
    with open(url, "rb") as f:
      return f.read()
  import os, hashlib, tempfile
  fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
  download_file(url, fp, skip_if_exists=not getenv("NOCACHE"))
  with open(fp, "rb") as f:
    return f.read()

def fetch_as_file(url):
  if url.startswith("/"):
    with open(url, "rb") as f:
      return f.read()
  import os, hashlib, tempfile
  fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
  download_file(url, fp, skip_if_exists=not getenv("NOCACHE"))
  return fp

def download_file(url, fp, skip_if_exists=True):
  import requests, os, pathlib
  if skip_if_exists and os.path.isfile(fp) and os.stat(fp).st_size > 0:
    return
  r = requests.get(url, stream=True)
  assert r.status_code == 200
  progress_bar = tqdm(total=int(r.headers.get('content-length', 0)), unit='B', unit_scale=True, desc=url)
  with tempfile.NamedTemporaryFile(dir=pathlib.Path(fp).parent, delete=False) as f:
    for chunk in r.iter_content(chunk_size=16384):
      progress_bar.update(f.write(chunk))
    f.close()
    os.rename(f.name, fp)


def my_unpickle(fb0):
  key_prelookup = defaultdict(list)
  def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    #print(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata)
    ident, storage_type, obj_key, location, obj_size = storage[0:5]
    assert ident == 'storage'
    assert prod(size) <= (obj_size - storage_offset)

    if storage_type not in [np.float16, np.float32]:
      if DEBUG: print(f"unsupported type {storage_type} on {obj_key} with shape {size}")
      ret = None
    else:
      ret = Tensor.empty(*size, dtype=dtypes.from_np(storage_type))
    key_prelookup[obj_key].append((storage_type, obj_size, ret, size, stride, storage_offset))
    return ret

  def _rebuild_parameter(*args):
    #print(args)
    pass

  class Dummy: pass

  class MyPickle(pickle.Unpickler):
    def find_class(self, module, name):
      #print(module, name)
      if name == 'FloatStorage': return np.float32
      if name == 'LongStorage': return np.int64
      if name == 'IntStorage': return np.int32
      if name == 'HalfStorage': return np.float16
      if module == "torch._utils":
        if name == "_rebuild_tensor_v2": return _rebuild_tensor_v2
        if name == "_rebuild_parameter": return _rebuild_parameter
      else:
        if module.startswith('pytorch_lightning'): return Dummy
        try:
          return super().find_class(module, name)
        except Exception:
          return Dummy

    def persistent_load(self, pid):
      return pid

  return MyPickle(fb0).load(), key_prelookup

def load_single_weight(t:Tensor, myfile, shape, strides, dtype, storage_offset, mmap_allowed=False):
  bytes_size = np.dtype(dtype).itemsize
  if t is None:
    myfile.seek(prod(shape) * bytes_size, 1)
    return

  bytes_offset = 0
  if storage_offset is not None:
    bytes_offset = storage_offset * bytes_size
    myfile.seek(bytes_offset)

  assert t.shape == shape or shape == tuple(), f"shape mismatch {t.shape} != {shape}"
  assert t.dtype.np == dtype and t.dtype.itemsize == bytes_size
  if any(s != 1 and st1 != st2 for s, st1, st2 in zip(shape, strides_for_shape(shape), strides)):
    # slow path
    buffer_size = sum(strides[i]*t.dtype.itemsize * (shape[i] - 1) for i in range(len(shape)))
    buffer_size += t.dtype.itemsize
    np_array = np.frombuffer(myfile.read(buffer_size), t.dtype.np)

    np_array = np.lib.stride_tricks.as_strided(
      np_array, shape=shape, strides=[i*t.dtype.itemsize for i in strides])

    lna = t.lazydata.op.arg
    lna.fxn = lambda _: np_array
    t.realize()
    return

  # ["METAL", "CLANG", "LLVM"] support readinto for more speed
  # ["GPU", "CUDA"] use _mmap since they have to copy in to the GPU anyway
  # this needs real APIs
  if t.device in ["METAL", "CLANG", "LLVM"]:
    del t.lazydata.op
    t.lazydata.realized = Device[t.lazydata.device].buffer(prod(t.shape), dtype=t.dtype)
    myfile.readinto(t.lazydata.realized._buffer())
  else:
    def _mmap(lna):
      assert myfile._compress_type == 0, "compressed data can't be mmaped"
      return np.memmap(myfile._fileobj._file, dtype=lna.dtype, mode='r', offset=myfile._orig_compress_start + bytes_offset, shape=lna.shape)
    def _read(lna):
      ret = np.empty(lna.shape, dtype=lna.dtype)
      myfile.readinto(ret.data)
      return ret
    if mmap_allowed and not OSX and t.device in ["GPU", "CUDA"]: t.lazydata.op.arg.fxn = _mmap
    else: t.lazydata.op.arg.fxn = _read
    t.realize()

def fake_torch_load_zipped(fb0, load_weights=True, multithreaded=True):
  if Device.DEFAULT in ["TORCH", "GPU", "CUDA"]: multithreaded = False  # multithreaded doesn't work with CUDA or TORCH. for GPU it's a wash with _mmap

  import zipfile
  with zipfile.ZipFile(fb0, 'r') as myzip:
    base_name = myzip.namelist()[0].split('/', 1)[0]
    with myzip.open(f'{base_name}/data.pkl') as myfile:
      ret = my_unpickle(myfile)
    if load_weights:
      def load_weight(k, vv):
        with myzip.open(f'{base_name}/data/{k}') as myfile:
          for v in vv:
            load_single_weight(v[2], myfile, v[3], v[4], v[0], v[5], mmap_allowed=True)
      if multithreaded:
        import concurrent.futures
        # 2 seems fastest
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
          futures = {executor.submit(load_weight, k, v):k for k,v in ret[1].items()}
          for future in (t:=tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
            if future.exception() is not None: raise future.exception()
            k = futures[future]
            t.set_description(f"loading {k} ram used: {GlobalCounters.mem_used/1e9:5.2f} GB")
      else:
        for k,v in (t := tqdm(ret[1].items())):
          t.set_description(f"loading {k} ram used: {GlobalCounters.mem_used/1e9:5.2f} GB")
          load_weight(k,v)
  return ret[0]

def fake_torch_load(b0):
  import io
  import struct

  # convert it to a file
  fb0 = io.BytesIO(b0)

  if b0[0:2] == b"\x50\x4b":
    return fake_torch_load_zipped(fb0)

  # skip three junk pickles
  pickle.load(fb0)
  pickle.load(fb0)
  pickle.load(fb0)

  ret, key_prelookup = my_unpickle(fb0)

  # create key_lookup
  key_lookup = pickle.load(fb0)
  key_real = [None] * len(key_lookup)
  for k,v in key_prelookup.items():
    assert len(v) == 1
    key_real[key_lookup.index(k)] = v[0]

  # read in the actual data
  for storage_type, obj_size, tensor, np_shape, np_strides, storage_offset in key_real:
    ll = struct.unpack("Q", fb0.read(8))[0]
    assert ll == obj_size, f"size mismatch {ll} != {obj_size}"
    assert storage_offset == 0, "not implemented"
    load_single_weight(tensor, fb0, np_shape, np_strides, storage_type, None)

  return ret

def get_child(parent, key):
  obj = parent
  for k in key.split('.'):
    if k.isnumeric():
      obj = obj[int(k)]
    elif isinstance(obj, dict):
      obj = obj[k]
    else:
      obj = getattr(obj, k)
  return obj
