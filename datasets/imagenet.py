# for imagenet download prepare.sh and run it
import glob, random
import json
import numpy as np
from PIL import Image
import functools, pathlib
import warnings 
warnings.filterwarnings('ignore')

BASEDIR = pathlib.Path(__file__).parent.parent / "datasets/imagenet"
ci = json.load(open(BASEDIR / "imagenet_class_index.json"))
cir = {v.split(",")[0]: k for k,v in ci.items()}
cir_to = {k: i for i, (k,v) in enumerate(ci.items())}

@functools.lru_cache(None)
def get_train_files():
  train_files = glob.glob(str(BASEDIR / "train/*/*"))
  return train_files

@functools.lru_cache(None)
def get_val_files():
  val_files = glob.glob(str(BASEDIR / "val/*/*"))
  return val_files

def normalization(image):
  input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
  input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
  image = image.permute([0,3,1,2]) / 255.0
  image -= input_mean
  image /= input_std
  return image

def rand_flip(image, axis=(1,2,3)):
  if random.random() <  1 / len(axis):
    image = np.flip(image, axis=axis).copy()
  return image

def random_scale(image, min_scale=0.8, max_scale=1.2):
    scale_factor = np.random.uniform(min_scale, max_scale)
    height, width = img_array.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    return image.resize((new_width, new_height))

def preprocess(image):
  image = normalization(image)
  if not val:
    image = random_scale(image)
    image, label = rand_flip(image)
  return image

#rrc = transforms.RandomResizedCrop(224)
import torchvision.transforms.functional as F
def image_load(fn):
  img = Image.open(fn).convert('RGB')
  img = F.resize(img, 256, Image.BILINEAR)
  img = F.center_crop(img, 224)
  ret = np.array(img)
  ret = preprocess(img)
  return ret

def iterate(bs=32, val=True, shuffle=True):
  files = get_val_files() if val else get_train_files()
  order = list(range(0, len(files)))
  if shuffle: random.shuffle(order)
  for i in range(0, len(files), bs):
    X = [image_load(files[i]) for i in order[i:i+bs]]
    Y = [cir_to[files[i].split("/")[-2]] for i in order[i:i+bs]]
    yield (np.array(X), np.array(Y))

def fetch_batch(bs, val=False):
  files = get_val_files() if val else get_train_files()
  samp = np.random.randint(0, len(files), size=(bs))
  files = [files[i] for i in samp]
  X = [image_load(x) for x in files]
  Y = [cir_to[x.split("/")[-2]] for x in files]
  print(cir_to.items())
  return np.array(X), np.array(Y)


if __name__ == "__main__":
  #X,Y = fetch_batch(64)
  #print(X.shape, Y)
  X,Y = fetch_batch(32,val=False)
  print(X.shape,Y)

