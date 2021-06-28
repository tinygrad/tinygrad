import os
import numpy as np
import pickle
import gzip

from datasets.utils import ImageDataset, download_from_url
from datasets import transforms as T

def download_mnist(root='./data', train=True, download=True):
  url_data = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz' if train else \
             'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
  url_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz' if train else \
               'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
  data_path = os.path.join(root, url_data.split('/')[-1])
  labels_path = os.path.join(root, url_labels.split('/')[-1])
  if download:
    if not os.path.exists(root): os.makedirs(root)
    if not os.path.exists(data_path):
      download_from_url(url_data, data_path)
      download_from_url(url_labels, labels_path)
    else: print(f'File {data_path} already downloaded.')
  if not os.path.exists(data_path): raise FileNotFoundError()
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  x = parse(data_path)[0x10:].reshape((-1, 1, 28, 28)).astype(np.float32) / 255.0 # x in [0, 1]
  y = parse(labels_path)[8:].astype(dtype=np.int32)
  return x, y, np.arange(10).astype(np.int32)

class MNIST(ImageDataset):
  sample_shape = (1, 28, 28)
  def __init__(self, root='./data', train=True, download=True, transform=lambda x: x, target_transform=lambda t: t):
    self.train = train
    x, y, classes = download_mnist(root=root, train=train, download=download)
    self.num_classes = len(classes)
    super().__init__(x, y, classes=classes, transform=transform, target_transform=target_transform)
