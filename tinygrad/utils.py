from hashlib import md5
import os

import numpy as np
import requests


def layer_init_uniform(m, h):
  ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)


def fetch_mnist():
  # Cache within session
  if hasattr(fetch_mnist, "data"):
    return fetch_mnist.data
  filename = "mnist.npz"
  url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/" + filename
  md5_hash = "8a61469f7ea1b51cbae51d4f78837e45"
  fp = os.path.join("/tmp", md5_hash)
  data = None
  # Cache across sessions
  if os.path.exists(fp):
    with open(fp, "rb") as f:
      data = f.read()
      if md5(data).hexdigest() != md5_hash:
        data = None
  if data is None:
    data = requests.get(url, timeout=10).content
    with open(fp + ".lock", "wb") as f:
      f.write(data)
    os.rename(fp + ".lock", fp)
  with np.load(fp, allow_pickle=True) as f:
    X_train, Y_train = f['x_train'], f['y_train']
    X_test, Y_test = f['x_test'], f['y_test']
  fetch_mnist.data = (X_train, Y_train, X_test, Y_test)
  return X_train, Y_train, X_test, Y_test
