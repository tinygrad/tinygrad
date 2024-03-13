import random, time, ctypes, struct
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
from tinygrad import dtypes, Tensor
from tinygrad.helpers import getenv, prod, Timing, Context
from multiprocessing import Queue, Process, shared_memory, connection, Lock

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

def shuffled_indices(n):
  indices = {}
  for i in range(n-1, -1, -1):
    j = random.randint(0, i)
    if i not in indices: indices[i] = i
    if j not in indices: indices[j] = j
    indices[i], indices[j] = indices[j], indices[i]
    yield indices[i]
    del indices[i]

def loader_process(q_in, q_out, X:Tensor):
  import signal
  signal.signal(signal.SIGINT, lambda _, __: exit(0))

  with Context(DEBUG=0):
    while (_recv := q_in.get()) is not None:
      idx, fn = _recv
      img = Image.open(fn)
      img = img.convert('RGB') if img.mode != "RGB" else img

      # eval: 76.08%, load in 0m7.366s (0m5.301s with simd)
      # sudo apt-get install libjpeg-dev
      # CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
      rescale = min(img.size) / 256
      crop_left = (img.width - 224*rescale) / 2.0
      crop_top = (img.height - 224*rescale) / 2.0
      img = img.resize((224, 224), Image.BILINEAR, box=(crop_left, crop_top, crop_left+224*rescale, crop_top+224*rescale))

      # broken out
      #img_tensor = Tensor(img.tobytes(), device='CPU')
      #storage_tensor = X[idx].contiguous().realize().lazydata.realized
      #storage_tensor._copyin(img_tensor.numpy())

      # faster
      X[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = img.tobytes()

      # ideal
      #X[idx].assign(img.tobytes())   # NOTE: this is slow!
      q_out.put(idx)

def batch_load_resnet(batch_size=64, val=False, shuffle=True):
  from extra.datasets.imagenet import get_train_files, get_val_files
  files = get_val_files() if val else get_train_files()
  from extra.datasets.imagenet import get_imagenet_categories
  cir = get_imagenet_categories()
  BATCH_COUNT = min(32, len(files) // batch_size)

  gen = shuffled_indices(len(files)) if shuffle else iter(range(len(files)))
  def enqueue_batch(num):
    for idx in range(num*batch_size, (num+1)*batch_size):
      fn = files[next(gen)]
      q_in.put((idx, fn))
      Y[idx] = cir[fn.split("/")[-2]]

  class Cookie:
    def __init__(self, num): self.num = num
    def __del__(self):
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
  shm = shared_memory.SharedMemory(name="resnet_X", create=True, size=prod(sz))
  procs = []

  try:
    # disk:shm is slower
    #X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:shm:{shm.name}")
    X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:/dev/shm/resnet_X")
    Y = [None] * (batch_size*BATCH_COUNT)

    for _ in range(64):
      p = Process(target=loader_process, args=(q_in, q_out, X))
      p.daemon = True
      p.start()
      procs.append(p)

    for bn in range(BATCH_COUNT): enqueue_batch(bn)

    # NOTE: this is batch aligned, last ones are ignored
    for _ in range(0, len(files)//batch_size): yield receive_batch()
  finally:
    # shutdown processes
    for _ in procs: q_in.put(None)
    for p in procs: p.join()
    shm.close()
    shm.unlink()

if __name__ == "__main__":
  from extra.datasets.imagenet import get_train_files, get_val_files
  VAL = getenv("VAL", 1)
  files = get_val_files() if VAL else get_train_files()
  with tqdm(total=len(files)) as pbar:
    for x,y,c in batch_load_resnet(val=VAL):
      pbar.update(x.shape[0])
