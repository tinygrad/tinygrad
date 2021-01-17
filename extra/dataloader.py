import numpy as np
import multiprocessing
import queue
from itertools import cycle

"""
Simple Dataloader for tinygrad.

Example usage:

from dataloader import DataLoader
import numpy as np

class Dataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return np.zeros((3, 32, 32)), 1


ds = Dataset(1024)
dl = DataLoader(ds, num_workers=4, batch_size=64)

x, y = next(dl)

print(x.shape)  # (64, 3, 32, 32)
print(y.shape)  # (64,)

"""

def default_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  if isinstance(batch[0], (int, float)):
    return np.array(batch)
  if isinstance(batch[0], (list, tuple)):
    return tuple(default_collate(var) for var in zip(*batch))


class NaiveDataLoader:
  def __init__(self, dataset, batch_size=64, collate_fn=default_collate):
    self.index = 0
    self.dataset = dataset
    self.batch_size = batch_size
    self.collate_fn = collate_fn

  def __iter__(self):
    self.index = 0
    return self

  def __next__(self):
    if self.index >= len(self.dataset):
        raise StopIteration
    batch_size = min(len(self.dataset) - self.batch_size, self.batch_size)
    return self.collate_fn([self.get() for _ in range(batch_size)])

  def get(self):
    item = self.dataset[self.index]
    self.index += 1
    return item


def worker_fn(dataset, index_queue, output_queue):
  while True:
    # Worker function, simply reads indices from index_queue, and adds the
    # dataset element to the output_queue
    try:
        index = index_queue.get(timeout=0)
    except queue.Empty:
        continue
    if index is None:
        break
    output_queue.put((index, dataset[index]))


class DataLoader(NaiveDataLoader):
  def __init__(
    self,
    dataset,
    batch_size=64,
    num_workers=1,
    prefetch_batches=2,
    collate_fn=default_collate,
  ):
    super().__init__(dataset, batch_size, collate_fn)
    self.num_workers = num_workers
    self.prefetch_batches = prefetch_batches
    self.output_queue = multiprocessing.Queue()
    self.index_queues = []
    self.workers = []
    self.worker_cycle = cycle(range(num_workers))
    self.cache = {}
    self.prefetch_index = 0

    for _ in range(num_workers):
        index_queue = multiprocessing.Queue()
        worker = multiprocessing.Process(
            target=worker_fn, args=(
                self.dataset, index_queue, self.output_queue)
        )
        worker.daemon = True
        worker.start()
        self.workers.append(worker)
        self.index_queues.append(index_queue)

    self.prefetch()
  def prefetch(self):
    while (
        self.prefetch_index < len(self.dataset)
        and self.prefetch_index
        < self.index + 2 * self.num_workers * self.batch_size
    ):
      # if the prefetch_index hasn't reached the end of the dataset
      # and it is not 2 batches ahead, add indexes to the index queues
      self.index_queues[next(self.worker_cycle)].put(self.prefetch_index)
      self.prefetch_index += 1

  def __iter__(self):
      self.index = 0
      self.cache = {}
      self.prefetch_index = 0
      self.prefetch()
      return self

  def get(self):
      self.prefetch()
      if self.index in self.cache:
        item = self.cache[self.index]
        del self.cache[self.index]
      else:
        while True:
          try:
            (index, data) = self.output_queue.get(timeout=0)
          except queue.Empty:  # output queue empty, keep trying
            continue
          if index == self.index:  # found our item, ready to return
            item = data
            break
          else:  # item isn't the one we want, cache for later
            self.cache[index] = data
      self.index += 1
      return item
  def __del__(self):
    try:
      for i, w in enumerate(self.workers):
        self.index_queues[i].put(None)
        w.join(timeout=5.0)

      for q in self.index_queues:
        q.cancel_join_thread()
        q.close()

      self.output_queue.cancel_join_thread()
      self.output_queue.close()
    finally:
      for w in self.workers:
        if w.is_alive():
          w.terminate()
