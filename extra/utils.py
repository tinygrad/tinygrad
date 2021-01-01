from tinygrad.tensor import Tensor

def fetch(url):
  import requests, os, hashlib, tempfile
  fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp) and os.stat(fp).st_size > 0:
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    print("fetching %s" % url)
    dat = requests.get(url).content
    with open(fp+".tmp", "wb") as f:
      f.write(dat)
    os.rename(fp+".tmp", fp)
  return dat

def get_parameters(obj):
  parameters = []
  if isinstance(obj, Tensor):
    parameters.append(obj)
  elif isinstance(obj, list):
    for x in obj:
      parameters.extend(get_parameters(x))
  elif hasattr(obj, '__dict__'):
    for k,v in obj.__dict__.items():
      parameters.extend(get_parameters(v))
  return parameters

def fake_torch_load(b0):
  import io
  import pickle
  import struct
  import numpy as np

  # convert it to a file
  fb0 = io.BytesIO(b0)

  # skip three junk pickles
  pickle.load(fb0)
  pickle.load(fb0)
  pickle.load(fb0)

  key_prelookup = {}

  class HackTensor:
    def __new__(cls, *args):
      #print(args)
      ident, storage_type, obj_key, location, obj_size, view_metadata = args[0]
      assert ident == 'storage'

      ret = np.zeros(obj_size, dtype=storage_type)
      key_prelookup[obj_key] = (storage_type, obj_size, ret, args[2], args[3])
      return ret

  class MyPickle(pickle.Unpickler):
    def find_class(self, module, name):
      #print(module, name)
      if name == 'FloatStorage':
        return np.float32
      if name == 'LongStorage':
        return np.int64
      if module == "torch._utils" or module == "torch":
        return HackTensor
      else:
        return pickle.Unpickler.find_class(self, module, name)

    def persistent_load(self, pid):
      return pid

  ret = MyPickle(fb0).load()

  # create key_lookup
  key_lookup = pickle.load(fb0)
  key_real = [None] * len(key_lookup)
  for k,v in key_prelookup.items():
    key_real[key_lookup.index(k)] = v

  # read in the actual data
  for storage_type, obj_size, np_array, np_shape, np_strides in key_real:
    ll = struct.unpack("Q", fb0.read(8))[0]
    assert ll == obj_size
    bytes_size = {np.float32: 4, np.int64: 8}[storage_type]
    mydat = fb0.read(ll * bytes_size)
    np_array[:] = np.frombuffer(mydat, storage_type)
    np_array.shape = np_shape

    # numpy stores its strides in bytes
    real_strides = tuple([x*bytes_size for x in np_strides])
    np_array.strides = real_strides

  return ret
