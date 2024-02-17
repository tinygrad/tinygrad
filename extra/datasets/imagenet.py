# for imagenet download prepare.sh and run it
import glob, random, json, math
import numpy as np
from PIL import Image
import functools, pathlib
from tinygrad.helpers import diskcache, getenv

BASEDIR = pathlib.Path(__file__).parent / "imagenet"

@functools.lru_cache(None)
def get_imagenet_categories():
  ci = json.load(open(BASEDIR / "imagenet_class_index.json"))
  return {v[0]: int(k) for k,v in ci.items()}

@diskcache
def _get_train_files(): return glob.glob(str(BASEDIR / "train/*/*"))
@functools.lru_cache(None)
def get_train_files():
  train_files = _get_train_files()
  # test train with less categories
  if getenv("TEST_CATS", 1000) != 1000:
    ci = get_imagenet_categories()
    train_files = [fn for fn in train_files if ci[fn.split("/")[-2]] < getenv("TEST_CATS", 1000)]
    print(f"Limiting to {getenv('TEST_CATS')} categories")
  if getenv("TEST_TRAIN"): train_files=train_files[:getenv("TEST_TRAIN")]
  print(f"Training on {len(train_files)} images")
  return train_files

@functools.lru_cache(None)
def get_val_files():
  val_files = glob.glob(str(BASEDIR / "val/*/*"))
  if getenv("TEST_CATS"):
    ci = get_imagenet_categories()
    val_files = [fn for fn in val_files if ci[fn.split("/")[-2]] < getenv("TEST_CATS")]
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
    img = np.flip(img, axis=1).copy()
  return img

# we don't use supplied imagenet bounding boxes, so scale min is just min_object_covered
# https://github.com/tensorflow/tensorflow/blob/e193d8ea7776ef5c6f5d769b6fb9c070213e737a/tensorflow/core/kernels/image/sample_distorted_bounding_box_op.cc
def random_resized_crop(img, size, scale=(0.10, 1.0), ratio=(3/4, 4/3)):
  w, h = img.size
  area = w * h

  # Crop
  random_solution_found = False
  for _ in range(10):
    aspect_ratio = random.uniform(ratio[0], ratio[1])
    max_scale = min(min(w * aspect_ratio / h, h / aspect_ratio / w), scale[1])
    target_area = area * random.uniform(scale[0], max_scale)

    w_new = int(round(math.sqrt(target_area * aspect_ratio)))
    h_new = int(round(math.sqrt(target_area / aspect_ratio)))

    if 0 < w_new <= w and 0 < h_new <= h:
        crop_left = random.randint(0, w - w_new + 1)
        crop_top = random.randint(0, h - h_new + 1)

        img = img.crop((crop_left, crop_top, crop_left + w_new, crop_top + h_new))
        random_solution_found = True
        break

  if not random_solution_found:
    # Center crop
    rescale = min(img.size) / 256
    crop_left = (img.width - 224 * rescale) / 2.0
    crop_top = (img.height - 224 * rescale) / 2.0
    img = img.resize((224, 224), Image.BILINEAR, box=(crop_left, crop_top, crop_left + 224 * rescale, crop_top + 224 * rescale))
  else:
    # Resize
    img = img.resize([size, size], Image.BILINEAR)

  return img

def preprocess_train(img):
  img = random_resized_crop(img, 224)
  img = rand_flip(np.array(img))
  #img = normalization(img)
  return img
