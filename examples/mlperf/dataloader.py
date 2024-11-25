import os, random, pickle, queue
from typing import List
from pathlib import Path
from multiprocessing import Queue, Process, shared_memory, connection, Lock, cpu_count

import numpy as np
from tinygrad import dtypes, Tensor
from tinygrad.helpers import getenv, prod, Context, round_up, tqdm

### ResNet

class MyQueue:
  def __init__(self, multiple_readers=True, multiple_writers=True):
    self._reader, self._writer = connection.Pipe(duplex=False)
    self._rlock = Lock() if multiple_readers else None
    self._wlock = Lock() if multiple_writers else None
  def get(self):
    if self._rlock: self._rlock.acquire()
    ret = pickle.loads(self._reader.recv_bytes())
    if self._rlock: self._rlock.release()
    return ret
  def put(self, obj):
    if self._wlock: self._wlock.acquire()
    self._writer.send_bytes(pickle.dumps(obj))
    if self._wlock: self._wlock.release()

def shuffled_indices(n, seed=None):
  rng = random.Random(seed)
  indices = {}
  for i in range(n-1, -1, -1):
    j = rng.randint(0, i)
    if i not in indices: indices[i] = i
    if j not in indices: indices[j] = j
    indices[i], indices[j] = indices[j], indices[i]
    yield indices[i]
    del indices[i]

def loader_process(q_in, q_out, X:Tensor, seed):
  import signal
  signal.signal(signal.SIGINT, lambda _, __: exit(0))

  from extra.datasets.imagenet import center_crop, preprocess_train
  from PIL import Image

  with Context(DEBUG=0):
    while (_recv := q_in.get()) is not None:
      idx, fn, val = _recv
      if fn is not None:
        img = Image.open(fn)
        img = img.convert('RGB') if img.mode != "RGB" else img

        if val:
          # eval: 76.08%, load in 0m7.366s (0m5.301s with simd)
          # sudo apt-get install libjpeg-dev
          # CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
          img = center_crop(img)
          img = np.array(img)
        else:
          # reseed rng for determinism
          if seed is not None:
            np.random.seed(seed * 2 ** 10 + idx)
            random.seed(seed * 2 ** 10 + idx)
          img = preprocess_train(img)
      else:
        # pad data with training mean
        img = np.tile(np.array([[[123.68, 116.78, 103.94]]], dtype=np.uint8), (224, 224, 1))

      # broken out
      #img_tensor = Tensor(img.tobytes(), device='CPU')
      #storage_tensor = X[idx].contiguous().realize().lazydata.base.realized
      #storage_tensor._copyin(img_tensor.numpy())

      # faster
      X[idx].contiguous().realize().lazydata.base.realized.as_buffer(force_zero_copy=True)[:] = img.tobytes()

      # ideal
      #X[idx].assign(img.tobytes())   # NOTE: this is slow!
      q_out.put(idx)
    q_out.put(None)

def batch_load_resnet(batch_size=64, val=False, shuffle=True, seed=None, pad_first_batch=False):
  from extra.datasets.imagenet import get_train_files, get_val_files
  files = get_val_files() if val else get_train_files()
  from extra.datasets.imagenet import get_imagenet_categories
  cir = get_imagenet_categories()

  if pad_first_batch:
    FIRST_BATCH_PAD = round_up(len(files), batch_size) - len(files)
  else:
    FIRST_BATCH_PAD = 0
  file_count = FIRST_BATCH_PAD + len(files)
  BATCH_COUNT = min(32, file_count // batch_size)

  def _gen():
    for _ in range(FIRST_BATCH_PAD): yield -1
    yield from shuffled_indices(len(files), seed=seed) if shuffle else iter(range(len(files)))
  gen = iter(_gen())

  def enqueue_batch(num):
    for idx in range(num*batch_size, (num+1)*batch_size):
      fidx = next(gen)
      if fidx != -1:
        fn = files[fidx]
        q_in.put((idx, fn, val))
        Y[idx] = cir[fn.split("/")[-2]]
      else:
        # padding
        q_in.put((idx, None, val))
        Y[idx] = -1

  shutdown = False
  class Cookie:
    def __init__(self, num): self.num = num
    def __del__(self):
      if not shutdown:
        try: enqueue_batch(self.num)
        except StopIteration: pass

  gotten = [0]*BATCH_COUNT
  def receive_batch():
    while 1:
      num = q_out.get()//batch_size
      gotten[num] += 1
      if gotten[num] == batch_size: break
    gotten[num] = 0
    return X[num*batch_size:(num+1)*batch_size], Y[num*batch_size:(num+1)*batch_size], Cookie(num)

  #q_in, q_out = MyQueue(multiple_writers=False), MyQueue(multiple_readers=False)
  q_in, q_out = Queue(), Queue()

  sz = (batch_size*BATCH_COUNT, 224, 224, 3)
  if os.path.exists("/dev/shm/resnet_X"): os.unlink("/dev/shm/resnet_X")
  shm = shared_memory.SharedMemory(name="resnet_X", create=True, size=prod(sz))
  procs = []

  try:
    # disk:shm is slower
    #X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:shm:{shm.name}")
    X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:/dev/shm/resnet_X")
    Y = [None] * (batch_size*BATCH_COUNT)

    for _ in range(cpu_count()):
      p = Process(target=loader_process, args=(q_in, q_out, X, seed))
      p.daemon = True
      p.start()
      procs.append(p)

    for bn in range(BATCH_COUNT): enqueue_batch(bn)

    # NOTE: this is batch aligned, last ones are ignored unless pad_first_batch is True
    for _ in range(0, file_count//batch_size): yield receive_batch()
  finally:
    shutdown = True
    # empty queues
    for _ in procs: q_in.put(None)
    q_in.close()
    for _ in procs:
      while q_out.get() is not None: pass
    q_out.close()
    # shutdown processes
    for p in procs: p.join()
    shm.close()
    try:
      shm.unlink()
    except FileNotFoundError:
      # happens with BENCHMARK set
      pass

### BERT

def process_batch_bert(data: List[dict]) -> dict[str, Tensor]:
  return {
    "input_ids": Tensor(np.concatenate([s["input_ids"] for s in data], axis=0), dtype=dtypes.float32),
    "input_mask": Tensor(np.concatenate([s["input_mask"] for s in data], axis=0), dtype=dtypes.default_float),
    "segment_ids": Tensor(np.concatenate([s["segment_ids"] for s in data], axis=0), dtype=dtypes.float32),
    "masked_lm_positions": Tensor(np.concatenate([s["masked_lm_positions"] for s in data], axis=0), dtype=dtypes.float32),
    "masked_lm_ids": Tensor(np.concatenate([s["masked_lm_ids"] for s in data], axis=0), dtype=dtypes.float32),
    "masked_lm_weights": Tensor(np.concatenate([s["masked_lm_weights"] for s in data], axis=0), dtype=dtypes.float32),
    "next_sentence_labels": Tensor(np.concatenate([s["next_sentence_labels"] for s in data], axis=0), dtype=dtypes.float32),
  }

def load_file(file: str):
  with open(file, "rb") as f:
    return pickle.load(f)

class InterleavedDataset:
  def __init__(self, files:List[str], cycle_length:int):
    self.dataset = files
    self.cycle_length = cycle_length
    self.queues = [queue.Queue() for _ in range(self.cycle_length)]
    for i in range(len(self.queues)): self.queues[i].queue.extend(load_file(self.dataset.pop(0)))
    self.queue_pointer = len(self.queues) - 1

  def get(self):
    # Round-robin across queues
    try:
      self.advance()
      return self.queues[self.queue_pointer].get_nowait()
    except queue.Empty:
      self.fill(self.queue_pointer)
      return self.get()

  def advance(self):
    self.queue_pointer = (self.queue_pointer + 1) % self.cycle_length

  def fill(self, queue_index: int):
    try:
      file = self.dataset.pop(0)
    except IndexError:
      return
    self.queues[queue_index].queue.extend(load_file(file))

# Reference: https://github.com/mlcommons/training/blob/1c8a098ae3e70962a4f7422c0b0bd35ae639e357/language_model/tensorflow/bert/run_pretraining.py, Line 394
def batch_load_train_bert(BS:int):
  from extra.datasets.wikipedia import get_wiki_train_files
  fs = sorted(get_wiki_train_files())
  train_files = []
  while fs: # TF shuffle
    random.shuffle(fs)
    train_files.append(fs.pop(0))

  cycle_length = min(getenv("NUM_CPU_THREADS", min(os.cpu_count(), 8)), len(train_files))
  assert cycle_length > 0, "cycle_length must be greater than 0"

  dataset = InterleavedDataset(train_files, cycle_length)
  buffer = [dataset.get() for _ in range(1000)]
  while True:
    batch = []
    for _ in range(BS):
      index = random.randint(0, 999)
      batch.append(buffer[index])
      buffer[index] = dataset.get()
    yield process_batch_bert(batch)

# Reference: https://github.com/mlcommons/training/blob/1c8a098ae3e70962a4f7422c0b0bd35ae639e357/language_model/tensorflow/bert/run_pretraining.py, Line 416
def batch_load_val_bert(BS:int):
  file =  getenv("BASEDIR", Path(__file__).parent.parents[1] / "extra" / "datasets" / "wiki") / "eval.pkl"
  dataset = load_file(file)
  idx = 0
  while True:
    start_idx = (idx * BS) % len(dataset)
    end_idx = ((idx + 1) * BS) % len(dataset)
    if start_idx < end_idx:
      yield process_batch_bert(dataset[start_idx:end_idx])
    else:  # wrap around the end to the beginning of the dataset
      yield process_batch_bert(dataset[start_idx:] + dataset[:end_idx])
    idx += 1

### UNET3D

def load_unet3d_data(preprocessed_dataset_dir, seed, queue_in, queue_out, X:Tensor, Y:Tensor):
  from extra.datasets.kits19 import rand_balanced_crop, rand_flip, random_brightness_augmentation, gaussian_noise

  while (data := queue_in.get()) is not None:
    idx, fn, val = data
    case_name = os.path.basename(fn).split("_x.npy")[0]
    x, y = np.load(preprocessed_dataset_dir / f"{case_name}_x.npy"), np.load(preprocessed_dataset_dir / f"{case_name}_y.npy")

    if not val:
      if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

      x, y = rand_balanced_crop(x, y)
      x, y = rand_flip(x, y)
      x, y = x.astype(np.float32), y.astype(np.uint8)
      x = random_brightness_augmentation(x)
      x = gaussian_noise(x)

    X[idx].contiguous().realize().lazydata.base.realized.as_buffer(force_zero_copy=True)[:] = x.tobytes()
    Y[idx].contiguous().realize().lazydata.base.realized.as_buffer(force_zero_copy=True)[:] = y.tobytes()

    queue_out.put(idx)
  queue_out.put(None)

def batch_load_unet3d(preprocessed_dataset_dir:Path, batch_size:int=6, val:bool=False, shuffle:bool=True, seed=None):
  assert preprocessed_dataset_dir is not None, "run preprocess_data on kits19"

  files = sorted(list(preprocessed_dataset_dir.glob("*_x.npy")))
  file_indices = list(range(len(files)))
  batch_count = min(32, len(files) // batch_size)

  queue_in, queue_out = Queue(), Queue()
  procs, data_out_count = [], [0] * batch_count
  shm_name_x, shm_name_y = "unet3d_x", "unet3d_y"
  sz = (batch_size * batch_count, 1, 128, 128, 128)
  if os.path.exists(f"/dev/shm/{shm_name_x}"): os.unlink(f"/dev/shm/{shm_name_x}")
  if os.path.exists(f"/dev/shm/{shm_name_y}"): os.unlink(f"/dev/shm/{shm_name_y}")
  shm_x = shared_memory.SharedMemory(name=shm_name_x, create=True, size=prod(sz))
  shm_y = shared_memory.SharedMemory(name=shm_name_y, create=True, size=prod(sz))

  shutdown = False
  class Cookie:
    def __init__(self, bc):
      self.bc = bc
    def __del__(self):
      if not shutdown:
        try: enqueue_batch(self.bc)
        except StopIteration: pass

  def enqueue_batch(bc):
    for idx in range(bc * batch_size, (bc+1) * batch_size):
      fn = files[next(ds_iter)]
      queue_in.put((idx, fn, val))

  def shuffle_indices(file_indices, seed=None):
    rng = random.Random(seed)
    rng.shuffle(file_indices)

  if shuffle: shuffle_indices(file_indices, seed=seed)
  ds_iter = iter(file_indices)

  try:
    X = Tensor.empty(*sz, dtype=dtypes.float32, device=f"disk:/dev/shm/{shm_name_x}")
    Y = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:/dev/shm/{shm_name_y}")

    for _ in range(cpu_count()):
      proc = Process(target=load_unet3d_data, args=(preprocessed_dataset_dir, seed, queue_in, queue_out, X, Y))
      proc.daemon = True
      proc.start()
      
      procs.append(proc)

    for bc in range(batch_count):
      enqueue_batch(bc)

    for _ in range(len(files) // batch_size):
      while True:
        bc = queue_out.get() // batch_size
        data_out_count[bc] += 1
        if data_out_count[bc] == batch_size: break

      data_out_count[bc] = 0
      yield X[bc * batch_size:(bc + 1) * batch_size], Y[bc * batch_size:(bc + 1) * batch_size], Cookie(bc)
  finally:
    shutdown = True

    for _ in procs: queue_in.put(None)
    queue_in.close()

    for _ in procs:
      while queue_out.get() is not None: pass
    queue_out.close()

    # shutdown processes
    for proc in procs: proc.join()

    shm_x.close()
    shm_y.close()
    try:
      shm_x.unlink()
      shm_y.unlink()
    except FileNotFoundError:
      # happens with BENCHMARK set
      pass

if __name__ == "__main__":
  def load_unet3d(val):
    assert not val, "validation set is not supported due to different sizes on inputs"

    from extra.datasets.kits19 import get_train_files, get_val_files, preprocess_dataset, TRAIN_PREPROCESSED_DIR, VAL_PREPROCESSED_DIR
    preprocessed_dir = VAL_PREPROCESSED_DIR if val else TRAIN_PREPROCESSED_DIR
    files = get_val_files() if val else get_train_files()

    if not preprocessed_dir.exists(): preprocess_dataset(files, preprocessed_dir, val)
    with tqdm(total=len(files)) as pbar:
      for x, _, _ in batch_load_unet3d(preprocessed_dir, val=val):
        pbar.update(x.shape[0])

  def load_resnet(val):
    from extra.datasets.imagenet import get_train_files, get_val_files
    files = get_val_files() if val else get_train_files()
    with tqdm(total=len(files)) as pbar:
      for x,y,c in batch_load_resnet(val=val):
        pbar.update(x.shape[0])

  load_fn_name = f"load_{getenv('MODEL', 'resnet')}"
  if load_fn_name in globals():
    globals()[load_fn_name](getenv("VAL", 1))
