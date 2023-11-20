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

def loader_process(q_in, q_out, X, Y):
  #import torchvision.transforms.functional as F
  from extra.datasets.imagenet import get_imagenet_categories
  cir = get_imagenet_categories()
  while (_recv := q_in.get()) is not None:
    idx, fn = _recv
    img = Image.open(fn)
    img = img.convert('RGB') if img.mode != "RGB" else img

    # 75.32%
    #factor = min(img.size)/256
    #l, t = ((img.size[0]/factor)-224)/2 * factor, ((img.size[1]/factor)-224)/2 * factor
    #img = img.transform((224, 224), Image.Transform.AFFINE, (factor, 0, l, 0, factor, t), Image.Resampling.BILINEAR)

    # 75.84% (non BILINEAR)
    # 76.02% with BILINEAR and rounded
    #factor = min(img.size)/256
    #print(l, t)
    #l, t = (img.size[0]-min(img.size))/2, (img.size[1]-min(img.size))/2
    #img = img.transform((256, 256), Image.Transform.EXTENT, (l, t, img.size[0]-l, img.size[1]-t), Image.Resampling.BILINEAR)
    #img = img.transform((256, 256), Image.Transform.AFFINE, (factor, 0, 0, 0, factor, 0), Image.Resampling.BILINEAR)
    #img = img.transform((int(img.size[0]/factor), int(img.size[1]/factor)), Image.Transform.EXTENT, (0, 0, img.size[0], img.size[1]), Image.Resampling.BILINEAR)
    #img = img.resize((int(img.width/factor), int(img.height/factor)), Image.Resampling.BILINEAR)
    #print(img.size, l, t)
    #l, t = int((img.width-224)/2+0.5), int((img.height-224)/2+0.5)
    #img = img.crop((l, t, l+224, t+224))

    # 76.14%
    #img = F.resize(img, 256, Image.BILINEAR)
    #img = F.center_crop(img, 224)

    # 76.14%
    factor = min(img.size)/256
    img = img.resize((int(img.width/factor), int(img.height/factor)), Image.Resampling.BILINEAR)
    crop_left = int(round((img.width - 224) / 2.0))
    crop_top = int(round((img.height - 224) / 2.0))
    img = img.crop((crop_left, crop_top, crop_left+224, crop_top+224))


    X[idx].assign(img.tobytes())
    Y[idx].assign(cir[fn.split("/")[-2]])
    q_out.put(idx)

def batch_load_resnet(batch_size=64, val=False, shuffle=True):
  from extra.datasets.imagenet import get_train_files, get_val_files
  files = get_val_files() if val else get_train_files()

  BATCH_COUNT = 10
  q_in, q_out = Queue(), Queue()
  X = Tensor.empty(batch_size*BATCH_COUNT, 224, 224, 3, dtype=dtypes.uint8, device=f"disk:/dev/shm/resnet_X")
  Y = Tensor.empty(batch_size*BATCH_COUNT, dtype=dtypes.uint32, device=f"disk:/dev/shm/resnet_Y")

  procs = []
  for _ in range(64):
    p = Process(target=loader_process, args=(q_in, q_out, X, Y))
    p.daemon = True
    p.start()
    procs.append(p)

  gen = shuffled_indices(len(files)) if shuffle else iter(range(len(files)))
  def enqueue_batch(num):
    for i in range(batch_size): q_in.put((num*batch_size+i, files[next(gen)]))
  for bn in range(BATCH_COUNT): enqueue_batch(bn)

  gotten = []
  def receive_batch(num):
    nonlocal gotten
    gotten, next_gotten = partition(gotten, lambda x: x >= num*batch_size and x < (num+1)*batch_size)
    while len(gotten) < batch_size:
      x = q_out.get()
      if x >= num*batch_size and x < (num+1)*batch_size: gotten.append(x)
      else: next_gotten.append(x)
    gotten = next_gotten
    return X[num*batch_size:(num+1)*batch_size], Y[num*batch_size:(num+1)*batch_size]

  # NOTE: this is batch aligned, last ones are ignored
  cbn = 0
  for _ in range(0, len(files)//batch_size):
    yield receive_batch(cbn)
    try:
      enqueue_batch(cbn)
    except StopIteration:
      pass
    cbn = (cbn+1) % BATCH_COUNT

  for _ in procs: q_in.put(None)
  for p in procs: p.join()

if __name__ == "__main__":
  from extra.datasets.imagenet import get_train_files, get_val_files
  VAL = True
  files = get_val_files() if VAL else get_train_files()
  with tqdm(total=len(files)) as pbar:
    for x,y in batch_load_resnet(val=VAL):
      pbar.update(x.shape[0])
