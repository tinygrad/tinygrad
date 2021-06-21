import requests
import os
import subprocess
import numpy as np
import pickle

from datasets.utils import Dataset

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def download_cifar100(root='./data', train=True, download=True):
  url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
  file_path = os.path.join(root, 'cifar-100-python.tar.gz')
  if download:
    if not os.path.exists(root):
      os.makedirs(root)
    if not os.path.exists(file_path):
      r = requests.get(url, allow_redirects=True)
      open(file_path, 'wb').write(r.content)
      cmd = f'tar -xf {file_path} -C {root}'
      process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
      output, error = process.communicate()
    else:
      print(f'File {file_path} already downloaded.')
  if not os.path.exists(file_path):
    raise FileNotFoundError()

  data_dict = unpickle(os.path.join(root, 'cifar-100-python', 'train' if train else 'test'))
  x = data_dict[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
  y = np.array(data_dict[b'fine_labels'], dtype=np.int32)
  classes = np.load(os.path.join(root, 'cifar-100-python', 'meta'), allow_pickle=True)['fine_label_names']
  return x, y, classes

class CIFAR100(Dataset):
  def __init__(self, root='./data', train=True, download=True, transform=lambda x: x, target_transform=lambda t: t):
    super().__init__()
    self.transform = transform
    self.target_transform = target_transform
    self.x, self.y, self.classes = download_cifar100(root=root, train=train, download=download)
    self.num_classes = len(self.classes)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    x = self.transform(self.x[idx])
    y = self.target_transform(self.y[idx])
    return x, y
