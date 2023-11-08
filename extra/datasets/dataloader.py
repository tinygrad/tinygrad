from typing import Any
import glob, random
import json
import numpy as np
from PIL import Image
import functools, pathlib
from simplejpeg import decode_jpeg
from multiprocessing import Pool
from functools import partial

BASEDIR = pathlib.Path(__file__).parent / "imagenet"
ci = json.load(open(BASEDIR / "imagenet_class_index.json"))
cir = {v[0]: int(k) for k,v in ci.items()}

@functools.lru_cache(None)
def get_train_files(dir=None):
  train_files = glob.glob(str(BASEDIR/"train/*/*") if not dir else str(dir/'train/*/*'))
  return train_files

@functools.lru_cache(None)
def get_val_files():
  val_files = glob.glob(str(BASEDIR / "val/*/*"))
  return val_files

import time
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop

def decode(fn):
  with open(fn, 'rb') as f:
    return decode_jpeg(f.read())

def get_transform(val):
  if not val:
    t = [
      transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
  else:
    t = [
     transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
     transforms.CenterCrop(224),
     transforms.ToTensor()
    ]
  return transforms.Compose(t)

toTensor = transforms.Compose([
    transforms.ToTensor()
])

def image_proc(fn,t):
  return t(Image.fromarray(decode(fn)))


def image_proc_timed(fn,t):
  s = time.perf_counter()
  X = t(Image.fromarray(decode(fn)))
  e = time.perf_counter() 
  return X, e-s

import math
def iterate(bs=16, val=False, shuffle=True, num_workers=16):
  files = get_val_files() if val else get_train_files()
  order = list(range(0, len(files)))
  if shuffle: random.shuffle(order)
  t = get_transform(val)
  with Pool(num_workers) as p:
    for i in range(0, len(files), bs)[:-1]:
      X = p.map(partial(image_proc,t=t), [files[j] for j in order[i:i + bs]], chunksize=math.ceil(bs/num_workers))
      Y = [cir[files[i].split("/")[-2]] for i in order[i:i+bs]]
      yield np.array(X),np.array(Y)

def benchmark_dataload_time():
  import statistics
  import os
  print('benchmarking dataload tm')
  all_trains,all_vals = 1281136,320284 
  epochs = 50
  tr = get_transform(False)
  t,u = [],[]
  DIR = pathlib.Path(__file__).parent/"imagenet"/"imagenette2"
  files = get_train_files(dir=DIR if os.path.exists(DIR) else None)
  order = list(range(0, len(files)))
  random.shuffle(order)
  stats = []
  for BS in [32,64,128]:
    for W in [4,8,16]:
      with Pool(W) as p:
        for _ in range(10):
          s = time.perf_counter()
          X,T = zip(*p.map(partial(image_proc_timed,t=tr), [files[j] for j in order[0:0+BS]], chunksize=BS//W))
          e = time.perf_counter()
          t.append(e-s)
          u.extend(T)
        print(f'**BS={BS} W={W}**')
        train_tm = (statistics.median(t))*(all_trains//BS)*epochs/(60*60)
        val_tm = (statistics.median(t))*(all_vals//BS)*(epochs//4)/(60*60)
        print(f'{train_tm+val_tm:7.2f} hrs total tm {train_tm:7.2f}hrs train tm {val_tm:7.2f}hrs val tm')
        print(f'mult: {(sum(t)/len(t))*1000:7.2f} avg read {statistics.median(t)*1000:7.2f} median read {max(t)*1000:7.2f} max read')
        print(f'unit: {(sum(u)/len(u))*1000:7.2f} avg read {statistics.median(u)*1000:7.2f} median read {max(u)*1000:7.2f} max read')
        stats.append((train_tm,BS,W))
  for i,(tt,BS,W) in enumerate(sorted(stats, key=lambda x:x[0])):
    opt_tm = (24*60*60*1000)/((all_trains//BS)*epochs+(all_vals//BS)*(epochs//4))
    print(f'RANK {i}: Under 24hrs={tt<=24} BS={BS} W={W} {tt:7.2f} hrs')
    if tt<=24:
      print(f'if GPU tm under {opt_tm:7.2f} ms, training will be under 24hrs')
  _, BS, W = sorted(stats, key=lambda x:x[0])[0]
  return BS,W
 
