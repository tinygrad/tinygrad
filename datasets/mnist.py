import requests
import os
import numpy as np
import pickle
import gzip

from datasets.utils import Dataset

def download_mnist(root='./data', train=True, download=True):
  if train:
    url_data = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
  else:
    url_data = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    url_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

  data_path = os.path.join(root, url_data.split('/')[-1])
  labels_path = os.path.join(root, url_labels.split('/')[-1])

  if download:
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(data_path):
        r = requests.get(url_data, allow_redirects=True)
        open(data_path, 'wb').write(r.content)
        r = requests.get(url_labels, allow_redirects=True)
        open(labels_path, 'wb').write(r.content)
    else:
        print(f'File {data_path} already downloaded.')
  if not os.path.exists(data_path):
    raise FileNotFoundError()

  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  x = parse(data_path)[0x10:].reshape((-1, 28 * 28)).astype(np.float32) / 255.0
  y = parse(labels_path)[8:].astype(dtype=np.int32)
  classes = np.arange(10).astype(np.int32)
  return x, y, classes

class MNIST(Dataset):
  def __init__(self, root='./data', train=True, download=True, transform=lambda x: x, target_transform=lambda t: t):
    super().__init__()
    self.train = train
    self.transform = transform
    self.target_transform = target_transform
    self.x, self.y, self.classes = download_mnist(root=root, train=train, download=download)
    self.num_classes = len(self.classes)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    x = self.transform(self.x[idx].reshape(1, 28, 28))
    y = self.target_transform(self.y[idx])
    return x, y
