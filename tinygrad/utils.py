import numpy as np

def mask_like(like, mask_inx, mask_value = 1.0):
  mask = np.zeros_like(like).reshape(-1)
  mask[mask_inx] = mask_value
  return mask.reshape(like.shape)

def layer_init_uniform(*x):
  ret = np.random.uniform(-1., 1., size=x)/np.sqrt(np.prod(x))
  return ret.astype(np.float32)

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

