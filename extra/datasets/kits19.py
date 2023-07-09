import random
import functools
from pathlib import Path
import requests
import numpy as np
import nibabel as nib
from scipy import signal
import torch
import torch.nn.functional as F
from tinygrad.tensor import Tensor

BASEDIR = Path(__file__).parent / "kits19" / "data"

"""
To download the dataset:
```sh
git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
cd ..
mv kits extra/datasets
```
"""

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
    image = F.interpolate(torch.from_numpy(np.expand_dims(image, axis=0)), size=new_shape, mode="trilinear", align_corners=True)
    label = F.interpolate(torch.from_numpy(np.expand_dims(label, axis=0)), size=new_shape, mode="nearest")
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

def iterate(val=True, shuffle=False):
  if not val: raise NotImplementedError
  files = get_val_files()
  order = list(range(0, len(files)))
  if shuffle: random.shuffle(order)
  for file in files:
    X, Y = preprocess(file)
    X = np.expand_dims(X, axis=0)
    yield (X, Y)

def gaussian_kernel(n, std):
  gaussian_1d = signal.gaussian(n, std)
  gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
  gaussian_3d = np.outer(gaussian_2d, gaussian_1d)
  gaussian_3d = gaussian_3d.reshape(n, n, n)
  gaussian_3d = np.cbrt(gaussian_3d)
  gaussian_3d /= gaussian_3d.max()
  return gaussian_3d

def pad_input(volume, roi_shape, strides, padding_mode="constant", padding_val=-2.2, dim=3):
  bounds = [(strides[i] - volume.shape[2:][i] % strides[i]) % strides[i] for i in range(dim)]
  bounds = [bounds[i] if (volume.shape[2:][i] + bounds[i]) >= roi_shape[i] else bounds[i] + strides[i] for i in range(dim)]
  paddings = [bounds[2]//2, bounds[2]-bounds[2]//2, bounds[1]//2, bounds[1]-bounds[1]//2, bounds[0]//2, bounds[0]-bounds[0]//2, 0, 0, 0, 0]
  return F.pad(torch.from_numpy(volume), paddings, mode=padding_mode, value=padding_val).numpy(), paddings

def sliding_window_inference(model, inputs, labels, roi_shape=(128, 128, 128), overlap=0.5):
  from tinygrad.jit import TinyJit
  mdl_run = TinyJit(lambda x: model(x).realize())
  image_shape, dim = list(inputs.shape[2:]), len(inputs.shape[2:])
  strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]
  bounds = [image_shape[i] % strides[i] for i in range(dim)]
  bounds = [bounds[i] if bounds[i] < strides[i] // 2 else 0 for i in range(dim)]
  inputs = inputs[
    ...,
    bounds[0]//2:image_shape[0]-(bounds[0]-bounds[0]//2),
    bounds[1]//2:image_shape[1]-(bounds[1]-bounds[1]//2),
    bounds[2]//2:image_shape[2]-(bounds[2]-bounds[2]//2),
  ]
  labels = labels[
    ...,
    bounds[0]//2:image_shape[0]-(bounds[0]-bounds[0]//2),
    bounds[1]//2:image_shape[1]-(bounds[1]-bounds[1]//2),
    bounds[2]//2:image_shape[2]-(bounds[2]-bounds[2]//2),
  ]
  inputs, paddings = pad_input(inputs, roi_shape, strides)
  padded_shape = inputs.shape[2:]
  size = [(inputs.shape[2:][i] - roi_shape[i]) // strides[i] + 1 for i in range(dim)]
  result = np.zeros((1, 3, *padded_shape), dtype=np.float32)
  norm_map = np.zeros((1, 3, *padded_shape), dtype=np.float32)
  norm_patch = gaussian_kernel(roi_shape[0], 0.125 * roi_shape[0])
  norm_patch = np.expand_dims(norm_patch, axis=0)
  for i in range(0, strides[0] * size[0], strides[0]):
    for j in range(0, strides[1] * size[1], strides[1]):
      for k in range(0, strides[2] * size[2], strides[2]):
        out = mdl_run(Tensor(inputs[..., i:roi_shape[0]+i,j:roi_shape[1]+j, k:roi_shape[2]+k])).numpy()
        result[..., i:roi_shape[0]+i, j:roi_shape[1]+j, k:roi_shape[2]+k] += out * norm_patch
        norm_map[..., i:roi_shape[0]+i, j:roi_shape[1]+j, k:roi_shape[2]+k] += norm_patch
  result /= norm_map
  result = result[..., paddings[4]:image_shape[0]+paddings[4], paddings[2]:image_shape[1]+paddings[2], paddings[0]:image_shape[2]+paddings[0]]
  return result, labels

if __name__ == "__main__":
  for X, Y in iterate():
    print(X.shape, Y.shape)
