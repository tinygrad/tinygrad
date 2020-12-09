import numpy as np
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

def get_parameters(model):
  if isinstance(model, Tensor):
    return [model]
  parameters = []
  if hasattr(model, '__dict__'):
    for k,v in model.__dict__.items():
      if isinstance(v, Tensor):
        parameters.append(v)
      elif isinstance(v, list):
        for x in v:
          parameters.extend(get_parameters(x))
      elif hasattr(v, '__dict__'):
        parameters.extend(get_parameters(v))
      #print(k, type(v))
  return parameters

