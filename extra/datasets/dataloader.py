import glob, random
import json
import numpy as np
from PIL import Image
import functools, pathlib
from simplejpeg import decode_jpeg
from multiprocessing import Pool
from functools import partial
from tinygrad.helpers import getenv
from queue import Queue
from threading import Thread
from tinygrad.helpers import getenv
import threading
import math

# TODO mem leak here - obj np.ndarray
class PreFetcher(Thread):
  def __init__(self,generator,max_prefetch=getenv("QS",1)):
    super().__init__()
    self.queue = Queue(1)
    self.generator = generator
    self.exit_event = threading.Event()
    self.daemon = True
    self.start()

  def run(self):
    try:
        for item in self.generator: 
          if self.exit_event.is_set(): 
            self.generator.close()
            del self.generator, self.queue
            return  # Stop prefetching if signaled to stop
          self.queue.put((True,item))
    except Exception as e:          self.queue.put((False,e))
    finally:                        
      if hasattr(self, 'queue'): self.queue.put((False,StopIteration))
  
  def stop(self):
    self.exit_event.set()

  def __next__(self):
    if not self.exit_event.is_set():
        success, next_item = self.queue.get()
        if success: 
          return next_item
        else:
            self.Continue = False
            raise next_item
    else: 
      raise StopIteration

  def __iter__(self): return self

BASEDIR = pathlib.Path(__file__).parent / "imagenet"
ci = json.load(open(BASEDIR / "imagenet_class_index.json"))
cir = {v[0]: int(k) for k,v in ci.items()}

@functools.lru_cache(None)
def get_train_files(dir=getenv("IMGNETTE2",0)):
  train_files = glob.glob(str(BASEDIR/"train/*/*") if not dir else str(pathlib.Path(__file__).parent/"imagenet"/"imagenette2"/'train/*/*'))
  return train_files

@functools.lru_cache(None)
def get_val_files(dir=getenv("IMGNETTE2",0)):
  val_files =  glob.glob(str(BASEDIR/"val/*/*") if not dir else str(pathlib.Path(__file__).parent/"imagenet"/"imagenette2"/'val/*/*'))
  return val_files

import time
from torchvision import transforms
import torchvision.transforms.functional as F

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
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
  return np.array(t(Image.fromarray(decode(fn))))

def image_proc_timed(fn,t):
  s = time.perf_counter()
  X = t(Image.fromarray(decode(fn)))
  e = time.perf_counter() 
  return np.array(X), e-s

def iterate(bs=16, val=False, shuffle=True, num_workers=16):
  files = get_val_files() if val else get_train_files()
  order = list(range(0, len(files)))
  if shuffle: random.shuffle(order)
  t = get_transform(val)
  with Pool(num_workers) as p:
    for i in range(0, len(files), bs)[:-1]:
      X = p.map(partial(image_proc,t=t),[files[j] for j in order[i:i + bs]],chunksize=math.ceil(bs/num_workers))
      Y = [cir[files[i].split("/")[-2]] for i in order[i:i+bs]]
      X,Y = np.array(X),np.array(Y)
      yield X, Y 

def benchmark_dataload_tm():
  import statistics
  from pathlib import Path
  import os
  if not os.path.exists(Path(__file__).parent/'imagenet'/'imagenette2'): 
    from extra.datasets.imagenet_download import get_imagenette2
    get_imagenette2()
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
  for BS in [16,32,64,128]:
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
        print(f'batch: {(sum(t)/len(t))*1000:7.2f} ms avg read {statistics.median(t)*1000:7.2f} ms median read {max(t)*1000:7.2f} ms max read')
        print(f'unit: {(sum(u)/len(u))*1000:7.2f} ms avg read {statistics.median(u)*1000:7.2f} ms median read {max(u)*1000:7.2f} ms max read')
        stats.append((train_tm,BS,W))
  for i,(tt,BS,W) in enumerate(sorted(stats, key=lambda x:x[0])):
    opt_tm = (24*60*60*1000)/((all_trains//BS)*epochs+(all_vals//BS)*(epochs//4))
    if tt<=24:
      note = (f'NOTE: if GPU tm under {opt_tm:7.2f} ms, then under 24hrs')
    else:
      note = (f'NOTE: no use in training, limited by cpu')
    print(f'RANK {i}: Under 24hrs={tt<=24} BS={BS} W={W} {tt:7.2f} hrs ' + note)
    
  _, BS, W = sorted(stats, key=lambda x:x[0])[0]
  return BS,W
 
if __name__ == '__main__':
  benchmark_dataload_tm()