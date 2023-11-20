import random
from PIL import Image
from tqdm import tqdm
from tinygrad.helpers import dtypes, partition
from tinygrad.tensor import Tensor, Device
from multiprocessing import Queue, Process

def shuffled_indices(n):
  indices = {}
  for i in range(n-1, -1, -1):
    j = random.randint(0, i)
    if i not in indices: indices[i] = i
    if j not in indices: indices[j] = j
    indices[i], indices[j] = indices[j], indices[i]
    yield indices[i]
    del indices[i]

def loader_process(q_in, q_out, X):
  while 1:
    idx, fn = q_in.get()
    img = Image.open(fn).convert('RGB')
    factor = min(img.size)/256
    img = img.resize((int(img.size[0]/factor), int(img.size[1]/factor)))
    l, t = (img.size[0]-224)//2, (img.size[1]-224)//2
    img = img.crop((l, t, l+224, t+224))
    X[idx].assign(img.tobytes())
    q_out.put(idx)

def batch_load_resnet(val=False):
  from extra.datasets.imagenet import get_train_files, get_val_files
  files = get_val_files() if val else get_train_files()
  print(f"imagenet files {len(files)}")
  gen = shuffled_indices(len(files))

  BATCH_COUNT = 10
  BS = 64
  q_in, q_out = Queue(), Queue()
  x_buf_shape = BS*BATCH_COUNT, 224, 224, 3
  X = Tensor.empty(*x_buf_shape, dtype=dtypes.uint8, device=f"disk:/dev/shm/resnet_X")

  for _ in range(64):
    p = Process(target=loader_process, args=(q_in, q_out, X))
    p.daemon = True
    p.start()

  def enqueue_batch(num):
    for i in range(BS): q_in.put((num*BS+i, files[next(gen)]))

  gotten = []
  def receive_batch(num):
    nonlocal gotten
    gotten, next_gotten = partition(gotten, lambda x: x >= num*BS and x < (num+1)*BS)
    while len(gotten) < BS:
      x = q_out.get()
      if x >= num*BS and x < (num+1)*BS: gotten.append(x)
      else: next_gotten.append(x)
    gotten = next_gotten
    return X[num*BS:(num+1)*BS]

  for bn in range(BATCH_COUNT): enqueue_batch(bn)
  cbn = 0
  with tqdm(total=len(files)) as pbar:
    # NOTE: this is batch aligned
    for _ in range(0, len(files)//BS):
      yield receive_batch(cbn)
      pbar.update(BS)
      try:
        enqueue_batch(cbn)
      except StopIteration:
        pass
      cbn = (cbn+1) % BATCH_COUNT

if __name__ == "__main__":
  for batch in batch_load_resnet(val=False):
    pass
    #print(batch.to(Device.DEFAULT))
