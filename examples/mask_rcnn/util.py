import os
import threading
import random
from queue import Queue, Empty
from typing import Callable, TypeVar, Optional, Tuple, Generator
import time

T = TypeVar('T')

# multi-threads file loading 
class FileLoader(threading.Thread):
  def __init__(
    self,
    dir: str,
    load_fn: Callable[[str], T],
    batch_size: int = 1,
    shuffle: bool = True,
    threads: int = 6,
    buffer_size: Optional[int] = None
  ) -> None:
    super().__init__()
    self.dir = dir
    self.load = load_fn
    self.threads = threads
    self.shuffle = shuffle
    if buffer_size is None:
      self.buffer_size = threads * 2
    else:
      self.buffer_size = buffer_size
    self.buffer: Queue[Tuple[str, T]] = Queue()
    print(f"Initializing file list for {dir} and {self.threads} threads...")
    self.file_lists = self._split_file_list()

    for _ in range(threads):
      threading.Thread(target=self._load_files, daemon=True).start()
    self.retrievals = 0

    
  def _initialize_file_list(self) -> list[str]:
    file_list = []
    for filename in os.listdir(self.dir):
      if os.path.isfile(os.path.join(self.dir, filename)):
        file_list.append(filename)
    return file_list

  def _split_file_list(self) -> list[list[str]]:
    file_list = self._initialize_file_list()
    self.num_files = len(file_list)
    print(f"Found {self.num_files} files in {self.dir}.")
    if self.shuffle:
      random.shuffle(file_list)
    # reduces read-time lock contention
    chunk_size = (len(file_list) + self.threads - 1) // self.threads
    return [file_list[i:i+chunk_size] for i in range(0, len(file_list), chunk_size)]

  def _load_files(self) -> None:
    thread_id = threading.get_ident() % self.threads
    while True:
      if len(self.buffer.queue) < self.buffer_size:
        if self.file_lists[thread_id]:
          filename = self.file_lists[thread_id].pop(0)
          filepath = os.path.join(self.dir, filename)
          converted_data = self.convert(filepath)
          self.buffer.put((filename, converted_data))
        else:
          break  # No more files to load
      else:
        # Sleep for a short while to avoid busy-waiting
        time.sleep(0.1)

  def get_next(self) -> Optional[Tuple[str, T]]:
      try:
          res = self.buffer.get_nowait()
          # TODO not threadsafe
          self.retrievals += 1
          return res
      except Empty:
          if self.retrievals < self.num_files:
              # TODO these metrics could be less noisy
              print("Buffer empty, waiting for next file...")
              res = self.buffer.get()
              # TODO not threadsafe
              self.retrievals += 1
              return res

  def get_batch(self) -> list[Tuple[str, T]]:
    batch = []
    for _ in range(self.batch_size):
      batch.append(self.get_next())
    return batch

  def __iter__(self) -> Generator[list[Tuple[str, T]], None, None]:
    while self.retrievals < self.num_files:
      yield self.get_batch()

if __name__ == '__main__':
  from PIL import Image
  # Example usage
  def convert_function(file_path: str) -> Image:
    return Image.open(file_path).convert("RGB")
    
  dir_path = '../../../coco/train2017'
  loader = FileLoader(dir_path, convert_function, buffer_size=10)
  
  time.sleep(10)

  while True:
    filename, data = loader.get_next()
    print(f"Got {filename} with shape {data.size}")
