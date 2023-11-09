# for imagenet download prepare.sh and run it
import glob, random
import math
import json
import time
import numpy as np
from PIL import Image
import functools, pathlib
from itertools import repeat
import threading
from queue import Queue
from threading import Thread
from tinygrad.helpers import getenv

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
def get_train_files():
  train_files = glob.glob(str(BASEDIR/"train/*/*"))
  return train_files

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

def image_resize(img, size, interpolation):
  w, h = img.size
  w_new = int((w / h) * size) if w > h else size
  h_new = int((h / w) * size) if h > w else size
  return img.resize([w_new, h_new], interpolation)

def rand_flip(img):
  if random.random() < 0.5:
    img = np.flip(img, axis=(0, 1)).copy()
  return img

def center_crop(img, size):
  w, h = img.size
  crop_h, crop_w = size, size
  crop_top = int(round((h - crop_h) / 2.0))
  crop_left = int(round((w - crop_w) / 2.0))
  return img.crop((crop_left, crop_top, size + crop_left, size + crop_top))

def random_resized_crop(img, size, scale=(0.08, 1.0), ratio=(3/4, 4/3)):
  w, h = img.size
  area = w * h

  # Crop
  log_ratio = [math.log(i) for i in ratio]
  random_solution_found = False
  for _ in range(10):
    target_area = area * random.uniform(scale[0], scale[1])
    aspect_ratio = math.exp(random.uniform(log_ratio[0], log_ratio[1]))

    w_new = int(round(math.sqrt(target_area * aspect_ratio)))
    h_new = int(round(math.sqrt(target_area / aspect_ratio)))

    if 0 < w_new <= w and 0 < h_new <= h:
        crop_left = random.randint(0, w - w_new + 1)
        crop_top = random.randint(0, h - h_new + 1)

        img = img.crop((crop_left, crop_top, crop_left + w_new, crop_top + h_new))
        random_solution_found = True
        break
    
  # Center crop
  if not random_solution_found:
    in_ratio = float(w) / float(h)
    if in_ratio < min(ratio):
        w_new = w
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h_new = h
        w = int(round(h * max(ratio)))
    else:
        w_new = w
        h_new = h
    crop_left = (h - h) // 2
    crop_top = (w - w) // 2
    img = img.crop((crop_left, crop_top, crop_left + w_new, crop_top + h_new))

  # Resize
  img = img.resize([size, size], Image.BILINEAR)

  return img

def preprocess(img, val):
  if not val:
    img = random_resized_crop(img, 224)
    img = rand_flip(np.array(img))
  else:
    img = center_crop(img, 224)
    img = np.array(img)
  #img = normalization(img)
  return img

def image_load(fn, val=True):
  #e1 = time.monotonic()
  img = Image.open(fn).convert('RGB')
  #e2 = time.monotonic()
  img = image_resize(img, 256, Image.BILINEAR)
  #e3 = time.monotonic()
  ret = preprocess(img, val)
  #e4 = time.monotonic()
  #print('at batch size 16, multi 16')
  #print(f'total {(e4-e1)*1000:7.2f} open {(e2-e1)*1000:7.2f} resize {(e3-e2)*1000:7.2f} preprocess {(e4-e2)*1000:7.2f}')
  return ret

from multiprocessing import Pool

def iterate(bs=16, val=True, shuffle=True, num_workers=0):
  files = get_val_files() if val else get_train_files()
  order = list(range(0, len(files)))
  if shuffle: random.shuffle(order)
  for i in range(0, len(files), bs)[:-1]:  # Don't get last batch so all batch shapes are consistent
    st = time.monotonic()
    if num_workers > 0:
      X = Pool(num_workers).starmap(image_load, zip([files[i] for i in order[i:i+bs]], repeat(val)))
    else:
      X = [image_load(files[i], val) for i in order[i:i+bs]]
    Y = [cir[files[i].split("/")[-2]] for i in order[i:i+bs]]
    et = time.monotonic()
    yield (np.array(X), np.array(Y), et - st)

def fetch_batch(bs, val=False):
  files = get_val_files() if val else get_train_files()
  samp = np.random.randint(0, len(files), size=(bs))
  files = [files[i] for i in samp]
  X = [image_load(x, val=val) for x in files]
  Y = [cir[x.split("/")[0]] for x in files]
  return np.array(X), np.array(Y)

if __name__ == "__main__":
  #X,Y = fetch_batch(64)
  #print(X.shape, Y)
  train_files = get_train_files()
  one_file = train_files[0]
  res = image_load(one_file)
  # fp32
  bytes = res.size * 4
  print(bytes)
  print(f'mb: {bytes/1000}')
  # total... not feasible to preprocess & save.
  print(f'total: {1.2*10**6*bytes/(1000*1000)}gb')

  for x,y,t in iterate(num_workers=0, val=False):
    print(f'one batch time {t*1000:7.2f} ms')
    print(f'pred one batch time {16*16} ms')



  # opt2 process on GPU
