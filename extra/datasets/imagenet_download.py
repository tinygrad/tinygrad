# Python version of https://gist.github.com/antoinebrl/7d00d5cb6c95ef194c737392ef7e476a
from tinygrad.helpers import fetch
from pathlib import Path
from tqdm import tqdm
import tarfile, os

IMAGENET_CLASS_INDEX_URL = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"

def imagenet_extract(file, path, small=False):
  with tarfile.open(name=file) as tar:
    if small: # Show progressbar only for big files
      for member in tar.getmembers(): tar.extract(path=path, member=member)
    else:
      for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())): tar.extract(path=path, member=member)
    tar.close()

def imagenet_prepare_val(imagenet_val_synset_labels_fn:str):
  # Read in the labels file
  with open(imagenet_val_synset_labels_fn, 'r') as f:
    labels = f.read().splitlines()
  f.close()
  # Get a list of images
  images = os.listdir(Path(__file__).parent / "imagenet" / "val")
  images.sort()
  # Create folders and move files into those
  for co,dir in enumerate(labels):
    os.makedirs(Path(__file__).parent / "imagenet" / "val" / dir, exist_ok=True)
    os.replace(Path(__file__).parent / "imagenet" / "val" / images[co], Path(__file__).parent / "imagenet" / "val" / dir / images[co])
  os.remove(imagenet_val_synset_labels_fn)

def imagenet_prepare_train():
  images = os.listdir(Path(__file__).parent / "imagenet" / "train")
  for co,tarf in enumerate(images):
    # for each tar file found. Create a folder with its name. Extract into that folder. Remove tar file
    if Path(Path(__file__).parent / "imagenet" / "train" / images[co]).is_file():
      images[co] = tarf[:-4] # remove .tar from extracted tar files
      os.makedirs(Path(__file__).parent / "imagenet" / "train" / images[co], exist_ok=True)
      imagenet_extract(Path(__file__).parent / "imagenet" / "train" / tarf, Path(__file__).parent/ "imagenet" / "train" / images[co], small=True)
      os.remove(Path(__file__).parent / "imagenet" / "train" / tarf)

if __name__ == "__main__":
  os.makedirs(Path(__file__).parent / "imagenet", exist_ok=True)
  os.makedirs(Path(__file__).parent / "imagenet" / "val", exist_ok=True)
  os.makedirs(Path(__file__).parent / "imagenet" / "train", exist_ok=True)
  fetch(IMAGENET_CLASS_INDEX_URL)
  imagenet_val_synset_labels_fn = fetch("https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt")
  imagenet_val_fn = fetch("https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar") # 7GB
  imagenet_extract(imagenet_val_fn, Path(__file__).parent / "imagenet" / "val")
  imagenet_prepare_val(imagenet_val_synset_labels_fn)
  if os.getenv('IMGNET_TRAIN', None) is not None:
    imagenet_train_fn = fetch("https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar") #138GB!
    imagenet_extract(imagenet_train_fn, Path(__file__).parent / "imagenet" / "train")
    imagenet_prepare_train()
