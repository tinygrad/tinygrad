import os
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

BASEDIR = "/Users/kafka/fun/imagenet"
train_files = open(os.path.join(BASEDIR, "train_files")).read().strip().split("\n")
val_files = open(os.path.join(BASEDIR, "val_files")).read().strip().split("\n")
ci = json.load(open(os.path.join(BASEDIR, "imagenet_class_index.json")))
cir = {v[0]: int(k) for k,v in ci.items()}

rrc = transforms.RandomResizedCrop(224)
def image_load(fn):
  img = Image.open(fn).convert('RGB')
  ret = np.array(rrc(img))
  return ret

def fetch_batch(bs, val=False):
  files = val_files if val else train_files
  samp = np.random.randint(0, len(files), size=(bs))
  files = [files[i] for i in samp]
  X = [image_load(os.path.join(BASEDIR, "val" if val else "train", x)) for x in files]
  Y = [cir[x.split("/")[0]] for x in files]
  return np.transpose(np.array(X), (0,3,1,2)), np.array(Y)

if __name__ == "__main__":
  X,Y = fetch_batch(64)
  print(X.shape, Y)

