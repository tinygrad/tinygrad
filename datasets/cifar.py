import os
import subprocess
import numpy as np
import pickle
from tqdm import tqdm

from datasets.utils import ImageDataset, download_from_url

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def download_cifar(root='./data', train=True, download=True, version='cifar-10'):
  assert version in ['cifar-10', 'cifar-100']
  url = f'https://www.cs.toronto.edu/~kriz/{version}-python.tar.gz'
  file_path = os.path.join(root, f'{version}-python.tar.gz')
  if download:
    if not os.path.exists(root): os.makedirs(root)
    if not os.path.exists(file_path):
      download_from_url(url, file_path)
      cmd = f'tar -xf {file_path} -C {root}'
      process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
      output, error = process.communicate()
    else: print(f'File {file_path} already downloaded.')
  if not os.path.exists(file_path): raise FileNotFoundError()
  if version == 'cifar-10':
    if train:
      x, y = [], []
      for i in range(1, 6):
        data_dict = unpickle(os.path.join(root, f'{version}-batches-py', f'data_batch_{i}'))
        x += [data_dict[b'data']]
        y += [data_dict[b'labels']]
      x, y = np.concatenate(x, 0), np.concatenate(y, 0)
    else:
      data_dict = unpickle(os.path.join(root, f'{version}-batches-py', 'test_batch'))
      x, y = data_dict[b'data'], data_dict[b'labels']
    classes = np.load(os.path.join(root, 'cifar-10-batches-py', 'batches.meta'), allow_pickle=True)['label_names']
  elif version == 'cifar-100':
    data_dict = unpickle(os.path.join(root, f'{version}-python', 'train' if train else 'test'))
    x, y = data_dict[b'data'], data_dict[b'fine_labels']
    classes = np.load(os.path.join(root, f'{version}-python', 'meta'), allow_pickle=True)['fine_label_names']
  return x.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0, np.array(y, dtype=np.int32), classes # x in [0, 1]

class __CIFAR(ImageDataset):
  sample_shape = (3, 32, 32)
  def __init__(self, version, root='./data', train=True, download=True, transform=lambda x: x, 
               target_transform=lambda t: t):
    self.train = train
    x, y, classes = download_cifar(root=root, train=train, download=download, version=version)
    self.num_classes = len(classes)
    super().__init__(x, y, classes=classes, transform=transform, target_transform=target_transform)

class CIFAR10(__CIFAR): 
  def __init__(self, *args, **kwargs): super().__init__('cifar-10', *args, **kwargs)
class CIFAR100(__CIFAR): 
  def __init__(self, *args, **kwargs): super().__init__('cifar-100', *args, **kwargs)
