# for imagenet download prepare.sh and run it
import glob, random, json, math
import numpy as np
from PIL import Image
import functools, pathlib
from tinygrad.helpers import DEBUG, diskcache, getenv

BASEDIR = pathlib.Path(__file__).parent / "imagenet"

@functools.lru_cache(None)
def get_imagenet_categories():
  ci = json.load(open(BASEDIR / "imagenet_class_index.json"))
  return {v[0]: int(k) for k,v in ci.items()}

@functools.lru_cache(None)
def get_train_files():
  train_files = glob.glob(str(BASEDIR / "train/*/*"))
  if getenv("TEST_TRAIN"): train_files = train_files[:getenv("TEST_TRAIN")]
  print(f"Training on {len(train_files)} images")
  return train_files

@functools.lru_cache(None)
def get_val_files():
  val_files = glob.glob(str(BASEDIR / "val/*/*"))
  if getenv("TEST_VAL"): val_files=val_files[:getenv("TEST_VAL")]
  return val_files

def normalization(img, transpose=False):
  img = np.float32(img)
  input_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, -1)
  input_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, -1)
  img = img / 255.0
  img -= input_mean
  img /= input_std
  if transpose: img = img.transpose([2,0,1])
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
