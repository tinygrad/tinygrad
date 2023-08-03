# for imagenet download prepare.sh and run it
import glob, random
import json
import numpy as np
from PIL import Image
import functools, pathlib
from itertools import repeat

BASEDIR = pathlib.Path(__file__).parent / "imagenet"
ci = json.load(open(BASEDIR / "imagenet_class_index.json"))
cir = {v[0]: int(k) for k,v in ci.items()}

@functools.lru_cache(None)
def get_train_files():
  train_files = open(BASEDIR / "train_files").read().strip().split("\n")
  return [(BASEDIR / "train" / x) for x in train_files]

@functools.lru_cache(None)
def get_val_files():
  val_files = glob.glob(str(BASEDIR / "val/*/*"))
  return val_files

def normalization(img):
  img = np.float32(img)
  input_mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
  input_std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
  img = img.transpose([2,0,1]) / 255.0
  img -= input_mean
  img /= input_std
  return img

def rand_flip(img):
  if random.random() < 0.5:
    img = np.flip(img, axis=(0, 1)).copy()
  return img

import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop
def preprocess(img, val):
  if not val:
    rrc = RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3))
    img = rrc(img)
    img = rand_flip(np.array(img))
  else:
    img = F.center_crop(img, 224)
    img = np.array(img)
  img = normalization(img)
  return img

def image_load(fn, val=True):
  img = Image.open(fn).convert('RGB')
  img = F.resize(img, 256, Image.BILINEAR)
  ret = preprocess(img, val)
  return ret

def iterate(bs=32, val=True, shuffle=True):
  files = get_val_files() if val else get_train_files()
  order = list(range(0, len(files)))
  if shuffle: random.shuffle(order)
  from multiprocessing import Pool
  p = Pool(16)
  for i in range(0, len(files), bs):
    X = p.starmap(image_load, zip([files[i] for i in order[i:i+bs]], repeat(val)))
    Y = [cir[files[i].split("/")[-2]] for i in order[i:i+bs]]
    yield (np.array(X), np.array(Y))

def fetch_batch(bs, val=False):
  files = get_val_files() if val else get_train_files()
  samp = np.random.randint(0, len(files), size=(bs))
  files = [files[i] for i in samp]
  X = [image_load(x, val=val) for x in files]
  Y = [cir[x.split("/")[0]] for x in files]
  return np.array(X), np.array(Y)

if __name__ == "__main__":
  X,Y = fetch_batch(64)
  print(X.shape, Y)

