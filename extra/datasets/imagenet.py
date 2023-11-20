# for imagenet download prepare.sh and run it
import glob, random, json
import numpy as np
from PIL import Image
import functools, pathlib
from tinygrad.helpers import DEBUG, diskcache

BASEDIR = pathlib.Path(__file__).parent / "imagenet"

@functools.lru_cache(None)
def get_imagenet_categories():
  ci = json.load(open(BASEDIR / "imagenet_class_index.json"))
  return {v[0]: int(k) for k,v in ci.items()}

@diskcache
def get_train_files(): return glob.glob(str(BASEDIR / "train/*/*"))

@functools.lru_cache(None)
def get_val_files(): return glob.glob(str(BASEDIR / "val/*/*"))

def image_load(fn):
  import torchvision.transforms.functional as F
  img = Image.open(fn).convert('RGB')
  img = F.resize(img, 256, Image.BILINEAR)
  img = F.center_crop(img, 224)
  ret = np.array(img)
  return ret

def iterate(bs=32, val=True, shuffle=True):
  cir = get_imagenet_categories()
  files = get_val_files() if val else get_train_files()
  order = list(range(0, len(files)))
  if DEBUG >= 1: print(f"imagenet size {len(order)}")
  if shuffle: random.shuffle(order)
  from multiprocessing import Pool
  p = Pool(16)
  for i in range(0, len(files), bs):
    X = p.map(image_load, [files[i] for i in order[i:i+bs]])
    Y = [cir[files[i].split("/")[-2]] for i in order[i:i+bs]]
    yield (np.array(X), np.array(Y))

def fetch_batch(bs, val=False):
  cir = get_imagenet_categories()
  files = get_val_files() if val else get_train_files()
  samp = np.random.randint(0, len(files), size=(bs))
  files = [files[i] for i in samp]
  X = [image_load(x) for x in files]
  Y = [cir[x.split("/")[0]] for x in files]
  return np.array(X), np.array(Y)

if __name__ == "__main__":
  X,Y = fetch_batch(64)
  print(X.shape, Y)

