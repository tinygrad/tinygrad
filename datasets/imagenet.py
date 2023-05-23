# for imagenet download prepare.sh and run it
import os, glob, random
import json
import numpy as np
from PIL import Image
import functools
import torchvision.transforms as transforms

BASEDIR = "/Users/kafka/fun/imagenet"
ci = json.load(open(os.path.join(BASEDIR, "imagenet_class_index.json")))
cir = {v[0]: int(k) for k,v in ci.items()}

@functools.lru_cache(None)
def get_train_files():
  train_files = open(os.path.join(BASEDIR, "train_files")).read().strip().split("\n")
  return [os.path.join(BASEDIR, "train", x) for x in train_files]

@functools.lru_cache(None)
def get_val_files():
  #val_files = open(os.path.join(BASEDIR, "val_files")).read().strip().split("\n")
  val_files = glob.glob(os.path.join(BASEDIR, "val", "*", "*"))
  return val_files

#rrc = transforms.RandomResizedCrop(224)
import torchvision.transforms.functional as F
def image_load(fn):
  img = Image.open(fn).convert('RGB')
  img = F.resize(img, 256, Image.BILINEAR)
  img = F.center_crop(img, 224)
  ret = np.array(img)
  return ret

def iterate(bs=32, val=True, shuffle=True):
  files = get_val_files() if val else get_train_files()
  order = list(range(0, len(files)))
  if shuffle: random.shuffle(order)
  for i in range(0, len(files), bs):
    X = [image_load(files[i]) for i in order[i:i+bs]]
    Y = [cir[files[i].split("/")[-2]] for i in order[i:i+bs]]
    yield (np.array(X), np.array(Y))

def fetch_batch(bs, val=False):
  files = get_val_files() if val else get_train_files()
  samp = np.random.randint(0, len(files), size=(bs))
  files = [files[i] for i in samp]
  X = [image_load(x) for x in files]
  Y = [cir[x.split("/")[0]] for x in files]
  return np.array(X), np.array(Y)

if __name__ == "__main__":
  X,Y = fetch_batch(64)
  print(X.shape, Y)

