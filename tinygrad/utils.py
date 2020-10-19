from hashlib import md5
import os

import numpy as np
import requests
from filelock import FileLock


def layer_init_uniform(m, h):
  ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)


mnist_filename = "mnist.npz"
mnist_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/" + mnist_filename
mnist_md5_hash = "8a61469f7ea1b51cbae51d4f78837e45"
mnist_fp = os.path.join("/tmp", mnist_md5_hash)

mnist_lock = FileLock(mnist_fp + ".lock")


def fetch_mnist():
  # Cache within session
  if hasattr(fetch_mnist, "data"):
    return fetch_mnist.data
  with mnist_lock:
    download = True
    # Cache across sessions
    if os.path.exists(mnist_fp):
      with open(mnist_fp, "rb") as f:
        if md5(f.read()).hexdigest() == mnist_md5_hash:
          download = False
    if download:
      data = requests.get(mnist_url, timeout=10).content
      with open(mnist_fp, "wb") as f:
        f.write(data)
    with np.load(mnist_fp, allow_pickle=True) as f:
      X_train, Y_train = f['x_train'], f['y_train']
      X_test, Y_test = f['x_test'], f['y_test']
  fetch_mnist.data = (X_train, Y_train, X_test, Y_test)
  return X_train, Y_train, X_test, Y_test
