import numpy as np
import requests
from tqdm import tqdm
import time

def download_from_url(url, dst):
  r = requests.head(url, allow_redirects=True)
  status_code = r.status_code
  while status_code != 200:
    print('Waiting for response...')
    time.sleep(2.0)
    r = requests.head(url, allow_redirects=True)
    status_code = r.status_code
  file_size = int(r.headers["Content-Length"])
  pbar = tqdm(total=file_size, initial=0, unit='B', unit_scale=True, desc=f'Downloading {url.split("/")[-1]}')
  r = requests.get(url, allow_redirects=True, stream=True)
  with open(dst, 'ab') as f:
    for chunk in r.iter_content(chunk_size=1024):
      if chunk:
        f.write(chunk)
        pbar.update(1024)
  pbar.close()

class Dataset:
  def __len__(self): raise NotImplementedError()
  def __getitem__(self, idx): raise NotImplementedError()
  def dataloader(self, *args, **kwargs): return DataLoader(self, *args, **kwargs)

class ImageDataset(Dataset):
  def __init__(self, x, y, classes=None, transform=lambda x: x, target_transform=lambda t: t):
    super().__init__()
    self.x, self.y, self.classes = x, y, classes
    if classes is None: 
      self.classes = sorted(list(set(self.y.tolist())))
    self.transform, self.target_transform = transform, target_transform
  def __len__(self): return len(self.x)
  def __getitem__(self, idx): return self.transform(self.x[idx]), self.target_transform(self.y[idx])

class TensorDataset(Dataset):
  def __init__(self, *tensors): 
    super().__init__()
    self.tensors = tensors
  def __len__(self): return len(self.tensors[0])
  def __getitem__(self, idx): return [t[idx] for t in self.tensors]

class DataLoader:
  def __init__(self, ds, batch_size, shuffle=False, seed=35771, steps=None):
    # TODO: implement num_workers, for paralellization.
    self.dataset = ds
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.steps = steps
    num_batches = int(np.ceil(len(self.dataset) / self.batch_size))
    self.num_batches = num_batches if self.steps is None else min(num_batches, self.steps)
    self.idxs = np.arange(len(self.dataset))
    if self.shuffle:
      np.random.seed(seed)
      np.random.shuffle(self.idxs)

  def __collate(self, samples_list):
    """Pack list of samples in a single batch.
    """
    if isinstance(samples_list[0], tuple) or isinstance(samples_list[0], list):
      num_elem = len(samples_list[0])
      batch_dict = {k: [] for k in range(num_elem)}
      for s in samples_list:
        for k in range(num_elem):
          batch_dict[k] += [s[k]]

      for k, v in batch_dict.items():
        batch_dict[k] = np.stack(v, 0)
      batch = list(batch_dict.values())

    elif isinstance(samples_list[0], dict):
      keys = list(samples_list[0].keys())
      batch_dict = {k: [] for k in keys}
      for s in samples_list:
        for k in keys:
          batch_dict[k] += [s[k]]

      for k, v in batch_dict.items():
        batch_dict[k] = np.stack(v, 0)
      batch = batch_dict

    else:
      raise NotImplementedError()
    return batch

  def __getitem__(self, idx):
    """Return a batch.
    """
    if idx == self.num_batches:
      raise StopIteration
    samples_idxs = self.idxs[idx * self.batch_size: (idx + 1) * self.batch_size]
    samples = [self.dataset.__getitem__(s_idx) for s_idx in samples_idxs]
    batch = self.__collate(samples)
    if idx == self.num_batches - 1:
      np.random.shuffle(self.idxs)

    return batch

  def __len__(self):
    return self.num_batches
