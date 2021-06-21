import numpy as np

class Dataset:
  def __len__(self):
    raise NotImplementedError()

  def __getitem__(self, idx):
    raise NotImplementedError()

  def dataloader(self, *args, **kwargs):
    """Convert a dataset into its related dataloader.
    """
    return DataLoader(self, *args, **kwargs)

class DataLoader:
  def __init__(self, ds, batch_size, shuffle=False, seed=35771):
    # TODO: implement num_workers, for paralellization.
    self.dataset = ds
    self.batch_size = batch_size
    self.shuffle = shuffle

    self.num_batches = int(np.ceil(len(self.dataset) / self.batch_size))
    self.idxs = np.arange(len(self.dataset))
    if shuffle:
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
    samples_idxs = self.idxs[idx * self.batch_size: (idx + 1) * self.batch_size]
    samples = [self.dataset.__getitem__(s_idx) for s_idx in samples_idxs]
    batch = self.__collate(samples)
    if idx == self.num_batches - 1:
      np.random.shuffle(self.idxs)
    return batch

  def __len__(self):
    return self.num_batches
