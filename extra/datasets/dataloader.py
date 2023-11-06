import multiprocessing
import cloudpickle
from typing import Any
import glob, random
import json
import numpy as np
from itertools import repeat
from PIL import Image
import functools, pathlib
import cloudpickle
from simplejpeg import decode_jpeg
from multiprocessing import Pool
from functools import partial

BASEDIR = pathlib.Path(__file__).parent / "imagenet"
ci = json.load(open(BASEDIR / "imagenet_class_index.json"))
cir = {v[0]: int(k) for k,v in ci.items()}

@functools.lru_cache(None)
def get_train_files():
  train_files = glob.glob(str(BASEDIR/"train/*/*"))
  return train_files

@functools.lru_cache(None)
def get_val_files():
  val_files = glob.glob(str(BASEDIR / "val/*/*"))
  return val_files

import time
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

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
      transforms.Normalize(mean=mean, std=std)
    ]
  else:
    t = [
     transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
     transforms.CenterCrop(224),
     transforms.ToTensor()
    ]
  return transforms.Compose(t)

def image_proc_n(fn,t):
  s = time.perf_counter()
  img = Image.fromarray(decode(fn))
  X = t(img)
  e = time.perf_counter()
  return X, e-s

toTensor = transforms.Compose([
    transforms.ToTensor()
])

rrc = RandomResizedCrop(224)
def image_proc(fn, val=False, t=None):
  img = Image.fromarray(decode(fn))
  img = F.resize(img, 256, Image.BILINEAR,antialias=True)
  if val:
    img = F.center_crop(img,224)
  else:
    img = rrc.forward(img)
    if random.random() < 0.5:
      img = F.hflip(img)
  img = toTensor(img)
  img = F.normalize(img/255.0,mean,std)
  return img 

import math
def iterate(bs=16, val=False, shuffle=True, num_workers=16):
  files = get_val_files() if val else get_train_files()
  order = list(range(0, len(files)))
  if shuffle: random.shuffle(order)
  t = get_transform(val)
  with Pool(num_workers) as p:
    for i in range(0, len(files), bs)[:-1]:
      s = time.perf_counter()
      X = p.map(partial(image_proc_n,t=t), [files[j] for j in order[i:i + bs]], chunksize=math.ceil(bs/num_workers))
      X,T = zip(*X)
      e = time.perf_counter()
      proc_tm = e-s
      print(f'{proc_tm*1000:7.2f} proc tm {max(T)*1000:7.2f} worse read tm')
      Y = [cir[files[i].split("/")[-2]] for i in order[i:i+bs]]
      yield np.array(X),np.array(Y),proc_tm
  
def proc(itermaker, q) -> None:
  try:
    for x in itermaker(): q.put(x)
  except Exception as e:
    q.put(e)
  finally:
    q.put(None)
    q.close()

class _CloudpickleFunctionWrapper:
  def __init__(self, fn): self.fn = fn
  def __getstate__(self): return cloudpickle.dumps(self.fn)
  def __setstate__(self, pfn): self.fn = cloudpickle.loads(pfn)
  def __call__(self, *args, **kwargs) -> Any:  return self.fn(*args, **kwargs)

def cross_process(itermaker, maxsize=8):
  q: multiprocessing.Queue = multiprocessing.Queue(maxsize)
  p = multiprocessing.Process(target=proc, args=(_CloudpickleFunctionWrapper(itermaker), q))
  p.start()
  while True:
    ret = q.get()
    if isinstance(ret, Exception): raise ret
    elif ret is None: break
    else: yield ret

if __name__ == '__main__':
  import statistics
  all_ts = 1281136
  epochs = 54
  tr = get_transform(False)
  t,u = [],[]
  files = get_train_files()
  order = list(range(0, len(files)))
  random.shuffle(order)
  stats = []
  for BS in [64,128,256]:
    for W in [2,4,6,8]:
      with Pool(W) as p:
        for _ in range(30):
          s = time.perf_counter()
          X,T = zip(*p.map(partial(image_proc_n,t=tr), [files[j] for j in order[0:0+BS]], chunksize=BS//W))
          e = time.perf_counter()
          t.append(e-s)
          u.extend(T)
        print(f'**BS={BS} W={W}**')
        train_time = (statistics.median(t))*(all_ts//BS+1)*epochs/(60*60)
        print(f'total time: {train_time:7.2f} hrs')
        print(f'mult: {(sum(t)/len(t))*1000:7.2f} avg read {statistics.median(t)*1000:7.2f} median read {max(t)*1000:7.2f} max read')
        print(f'unit: {(sum(u)/len(u))*1000:7.2f} avg read {statistics.median(u)*1000:7.2f} median read {max(u)*1000:7.2f} max read')
        stats.append((train_time,BS,W))
  for i,(tt,BS,W) in enumerate(sorted(stats, key=lambda x:x[0])):
    print(f'RANK {i}: BS={BS} W={W} {tt} hrs')
 
