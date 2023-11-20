from tqdm import tqdm, trange
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, dtypes, prod
from multiprocessing import Queue, Process
import multiprocessing.shared_memory as shared_memory

def image_load(fn):
  #img = Image.open(fn).convert('RGB')
  """
  rsz = (min(img.size)/256) * 224
  t,l = (img.size[1]-rsz)/2, (img.size[0]-rsz)/2
  import torchvision.transforms.functional as F
  img = F.resized_crop(img, t, l, t+rsz, l+rsz, 224)
  """
  #print(img.size)

  #print(img.size, w, h)

  #img = img.resize((int(img.size[0] / factor), int(img.size[1] / factor)))

  #import torchvision.transforms.functional as F
  #img = F.resize(img, 256, Image.BILINEAR)
  #img = F.center_crop(img, 224)
  #print(img.size, factor)
  #return fn
  return None


def train_resnet():
  from extra.datasets.imagenet import get_train_files
  files = get_train_files()
  print(f"train files {len(files)}")
  gen = shuffled_indices(len(files))

  BS = 64
  q_in, q_out = Queue(), Queue()
  x_buf_shape = BS, 3, 224, 224

  X = Tensor.empty(*x_buf_shape, dtype=dtypes.uint8, device=f"disk:/dev/shm/resnet_X")
  p = Process(target=loader_thread, args=(q_in, q_out, X))
  p.daemon = True
  p.start()

  for i in range(BS):
    idx = next(gen)
    q_in.put((i, files[idx]))

  for i in range(BS):
    print(q_out.get())


  """
  for i in range(10):
    print(list(shuffled_indices(5000))[0:10])
  order = list(range(0, len(files)))
  random.shuffle(order)
  print("shuffled")
  file_order = [files[i] for i in order]
  print("shuffled files")

  """

  #for fn in tqdm(files):
  #  img = Image.open(fn).convert('RGB')

  """
  BS = 1
  from multiprocessing import Pool
  p = Pool(64)
  load_map = p.imap(image_load, file_order)
  #for _ in trange(0, len(files), BS):
  #  batch_x = [next(load_map) for _ in range(BS)]
  for _ in tqdm(load_map, total=len(files)):
    pass
  """
  #for _ in tqdm(files):
  #  pass

  """
  BS = 64
  i = 0
  with tqdm(total=60000) as pbar:
    X = p.map(image_load, [files[i] for i in order[i:i+BS]])
    i += BS
  """

  """
  from extra.helpers import cross_process
  BS = 64
  iterator = cross_process(lambda: iterate(BS))
  x,ny = next(iterator)
  dat = Tensor(x)
  with tqdm(total=60000) as pbar:
    while dat is not None:
      pbar.update(BS)
      try:
        x,ny = next(iterator)
        dat = Tensor(x)
      except StopIteration:
        dat = None
  """

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  # TODO: Unet3d
  pass

def train_rnnt():
  # TODO: RNN-T
  pass

def train_bert():
  # TODO: BERT
  pass

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()


