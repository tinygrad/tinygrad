# for kits19 clone the offical repo and run get_imaging.py
import random
import functools
from pathlib import Path
import pickle
import requests
import numpy as np
import nibabel as nib
import torch
from torch.nn.functional import interpolate

BASEDIR = Path(__file__).parent.parent.parent.resolve() / "kits19" / "data"

@functools.lru_cache(None)
def get_val_files():
  data = requests.get("https://raw.githubusercontent.com/mlcommons/training/master/image_segmentation/pytorch/evaluation_cases.txt")
  return sorted([x for x in BASEDIR.iterdir() if x.stem.split("_")[-1] in data.text.split("\n")])

def load_pair(file_path):
  image, label = nib.load(file_path / "imaging.nii.gz"), nib.load(file_path / "segmentation.nii.gz")
  image_spacings = image.header["pixdim"][1:4].tolist()
  image, label = image.get_fdata().astype(np.float32), label.get_fdata().astype(np.uint8)
  image, label = np.expand_dims(image, 0), np.expand_dims(label, 0)
  return image, label, image_spacings

def resample3d(image, label, image_spacings, target_spacing=(1.6, 1.2, 1.2)):
  if image_spacings != target_spacing:
    spc_arr, targ_arr, shp_arr = np.array(image_spacings), np.array(target_spacing), np.array(image.shape[1:])
    new_shape = (spc_arr / targ_arr * shp_arr).astype(int).tolist()
    image = interpolate(torch.from_numpy(np.expand_dims(image, axis=0)), size=new_shape, mode="trilinear", align_corners=True)
    label = interpolate(torch.from_numpy(np.expand_dims(label, axis=0)), size=new_shape, mode="nearest")
    image = np.squeeze(image.numpy(), axis=0)
    label = np.squeeze(label.numpy(), axis=0)
  return image, label

def normal_intensity(image, min_clip=-79.0, max_clip=304.0, mean=101.0, std=76.9):
  image = np.clip(image, min_clip, max_clip)
  image = (image - mean) / std
  return image

def pad_to_min_shape(image, label, roi_shape=(128, 128, 128)):
  current_shape = image.shape[1:]
  bounds = [max(0, roi_shape[i] - current_shape[i]) for i in range(3)]
  paddings = [(0, 0)] + [(bounds[i] // 2, bounds[i] - bounds[i] // 2) for i in range(3)]
  image = np.pad(image, paddings, mode="edge")
  label = np.pad(label, paddings, mode="edge")
  return image, label

def preprocess(file_path):
  image, label, image_spacings = load_pair(file_path)
  image, label = resample3d(image, label, image_spacings)
  image = normal_intensity(image.copy())
  image, label = pad_to_min_shape(image, label)
  return image, label

def load(fp):
  with fp.open("rb") as f:
    X, Y = pickle.load(f)
  return X, Y

def save(image, label, fp):
  image = image.astype(np.float32)
  label = label.astype(np.uint8)
  fp.parent.mkdir(exist_ok=True)
  with fp.open("wb") as f:
    pickle.dump([image, label], f)

def iterate(val=True, shuffle=True):
  if not val: raise NotImplementedError
  files = get_val_files()
  order = list(range(0, len(files)))
  if shuffle: random.shuffle(order)
  for file in files:
    preprocessed_file = BASEDIR.parent / "preprocessed" / f"{file.stem}.pkl"
    if preprocessed_file.is_file():
      X, Y = load(preprocessed_file)
    else:
      X, Y = preprocess(file)
      save(X, Y, preprocessed_file)
    yield (X, Y, file.stem)

if __name__ == "__main__":
  # preprocess all files
  for X, Y, case in iterate():
    print(case, X.shape, Y.shape)
