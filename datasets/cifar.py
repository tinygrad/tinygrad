import os
import subprocess
import numpy as np
import pickle
from tqdm import tqdm

from datasets.utils import Dataset, download_from_url

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def download_cifar(root='./data', train=True, download=True, version='cifar-10'):
  assert version in ['cifar-10', 'cifar-100']
  url = f'https://www.cs.toronto.edu/~kriz/{version}-python.tar.gz'
  file_path = os.path.join(root, f'{version}-python.tar.gz')
  if download:
    if not os.path.exists(root):
      os.makedirs(root)
    if not os.path.exists(file_path):
      download_from_url(url, file_path)
      cmd = f'tar -xf {file_path} -C {root}'
      process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
      output, error = process.communicate()
    else:
      print(f'File {file_path} already downloaded.')
  if not os.path.exists(file_path):
    raise FileNotFoundError()

  if version == 'cifar-10':
    if train:
      x, y = [], []
      for i in range(1, 6):
        filename = f'data_batch_{i}'
        data_dict = unpickle(os.path.join(root, f'{version}-batches-py', filename))
        x += [data_dict[b'data']]
        y += [data_dict[b'labels']]
      x = np.concatenate(x, 0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
      y = np.concatenate(y, 0).astype(np.int32)
    else:
      data_dict = unpickle(os.path.join(root, f'{version}-batches-py', 'test_batch'))
      x = data_dict[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
      y = np.array(data_dict[b'labels'], dtype=np.int32)
    classes = np.load(os.path.join(root, 'cifar-10-batches-py', 'batches.meta'), allow_pickle=True)['label_names']

  elif version == 'cifar-100':
    data_dict = unpickle(os.path.join(root, 'cifar-100-python', 'train' if train else 'test'))
    x = data_dict[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.array(data_dict[b'fine_labels'], dtype=np.int32)
    classes = np.load(os.path.join(root, 'cifar-100-python', 'meta'), allow_pickle=True)['fine_label_names']
  return x, y, classes

class __CIFAR(Dataset):
  def __init__(self, version, root='./data', train=True, download=True, transform=lambda x: x, target_transform=lambda t: t):
    super().__init__()
    self.transform = transform
    self.target_transform = target_transform
    self.x, self.y, self.classes = download_cifar(root=root, train=train, download=download, version=version)
    self.num_classes = len(self.classes)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    x = self.transform(self.x[idx])
    y = self.target_transform(self.y[idx])
    return x, y

class CIFAR10(__CIFAR):
  def __init__(self, root='./data', train=True, download=True, transform=lambda x: x, target_transform=lambda t: t):
    super().__init__('cifar-10', root=root, train=train, download=download, transform=transform, 
                     target_transform=target_transform)

class CIFAR100(__CIFAR):
  def __init__(self, root='./data', train=True, download=True, transform=lambda x: x, target_transform=lambda t: t):
    super().__init__('cifar-100', root=root, train=train, download=download, transform=transform, 
                     target_transform=target_transform)
