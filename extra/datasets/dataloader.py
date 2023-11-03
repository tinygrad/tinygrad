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

#rrc = transforms.RandomResizedCrop(224)
import torch
import time
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop, Normalize, RandomHorizontalFlip
import pyvips
from io import BytesIO

def decode(fn):
  with open(fn, 'rb') as f:
    return decode_jpeg(f.read())

rrc = RandomResizedCrop(224)
def image_load(fn, val=False):
  #s = time.perf_counter()
  #img = Image.open(fn).convert('RGB')
  img = Image.fromarray(decode(fn))
  img = F.resize(img, 256, Image.BILINEAR,antialias=True)
  
  #e = time.perf_counter()
 # load_t = e-s
  if val:
    img = F.center_crop(img,224)
  else:
  #  s1 = time.perf_counter()
    img = rrc.forward(img)
  #  e1 = time.perf_counter()
    if random.random() < 0.5:
      #rhf=RandomHorizontalFlip(p=0.5)
      #img=rhf.forward(img)
      img = F.hflip(img)
      #e=time.perf_counter()
      #s=time.perf_counter()
      #e=time.perf_counter()
    #print(f'load timne {load_t*1000:7.2f}ms norm {(e-s)*1000:7.2f}ms resize {(e1-s1)*1000:7.2f}ms flip {(e2-e1)*1000:7.2f}ms')
  return np.float32(img)

def iterate(bs=32, val=True, shuffle=True, num_workers=16):
  files = get_val_files() if val else get_train_files()
  order = list(range(0, len(files)))
  if shuffle: random.shuffle(order)
  from multiprocessing import Pool
  p = Pool(num_workers)
  for i in range(0, len(files), bs):
    X = p.starmap(image_load, zip([files[i] for i in order[i:i+bs]], repeat(val)))
    Y = [cir[files[i].split("/")[-2]] for i in order[i:i+bs]]
    yield (np.array(X), np.array(Y))
  
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

def cross_process(itermaker, maxsize=16):
  q: multiprocessing.Queue = multiprocessing.Queue(maxsize)
  # multiprocessing uses pickle which cannot dump lambdas, so use cloudpickle.
  p = multiprocessing.Process(target=proc, args=(_CloudpickleFunctionWrapper(itermaker), q))
  p.start()
  while True:
    ret = q.get()
    if isinstance(ret, Exception): raise ret
    elif ret is None: break
    else: yield ret

if __name__ == '__main__':
  import time
  r = []
  for i,f in enumerate(get_train_files()[:100]):
    s = time.monotonic()
    image_load(f)
    s1 = time.monotonic()  
    r.append(s1-s)
    if i != 0 and i%100==0:break
  print(f'{(sum(r)/len(r))*1000:7.2f}ms')