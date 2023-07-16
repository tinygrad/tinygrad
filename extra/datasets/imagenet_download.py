# Python version of https://gist.github.com/antoinebrl/7d00d5cb6c95ef194c737392ef7e476a
import tarfile, os
from pathlib import Path
from tqdm import tqdm
from extra.utils import download_file
from tinygrad.helpers import Files

BASE_PATH = Files.tempdir / "imagenet"

def imagenet_extract(file, path, small=False):
  with tarfile.open(name=file) as tar:
    if small: # Show progressbar only for big files
      for member in tar.getmembers(): tar.extract(path=path, member=member)
    else:
      for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())): tar.extract(path=path, member=member)
    tar.close()

def imagenet_prepare_val():
  # Read in the labels file
  with open(BASE_PATH / "imagenet_2012_validation_synset_labels.txt", 'r') as f:
    labels = f.read().splitlines()
  f.close()
  # Get a list of images
  images = os.listdir(BASE_PATH / "val")
  images.sort()
  # Create folders and move files into those
  for co,dir in enumerate(labels):
    os.makedirs(BASE_PATH / "val" / dir, exist_ok=True)
    os.replace(BASE_PATH / "val" / images[co], BASE_PATH / "val" / dir / images[co])
  os.remove(BASE_PATH / "imagenet_2012_validation_synset_labels.txt")

def imagenet_prepare_train():
  images = os.listdir(BASE_PATH / "train")
  for co,tarf in enumerate(images):
    # for each tar file found. Create a folder with its name. Extract into that folder. Remove tar file
    if Path(BASE_PATH / "train" / images[co]).is_file():
      images[co] = tarf[:-4] # remove .tar from extracted tar files
      os.makedirs(BASE_PATH / "train" / images[co], exist_ok=True)
      imagenet_extract(BASE_PATH / "train" / tarf, BASE_PATH / "imagenet" / "train" / images[co], small=True)
      os.remove(BASE_PATH / "train" / tarf)

if __name__ == "__main__":
  os.makedirs(BASE_PATH, exist_ok=True)
  os.makedirs(BASE_PATH / "val", exist_ok=True)
  os.makedirs(BASE_PATH / "train", exist_ok=True)
  download_file("https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json", BASE_PATH / "imagenet_class_index.json")
  download_file("https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt", BASE_PATH/ "imagenet_2012_validation_synset_labels.txt")
  download_file("https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar", BASE_PATH / "ILSVRC2012_img_val.tar") # 7GB
  imagenet_extract(BASE_PATH / "ILSVRC2012_img_val.tar", BASE_PATH / "val")
  imagenet_prepare_val()
  if os.getenv('IMGNET_TRAIN', None) is not None:
    download_file("https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar", BASE_PATH / "ILSVRC2012_img_train.tar") #138GB!
    imagenet_extract(BASE_PATH / "ILSVRC2012_img_train.tar", BASE_PATH / "train")
    imagenet_prepare_train()
