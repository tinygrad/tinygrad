import os
import random
import glob
import functools
from tqdm import tqdm
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import signal, ndimage
import torch
import torch.nn.functional as F
from tinygrad.tensor import Tensor
from tinygrad.helpers import fetch, getenv

BASEDIR = Path(__file__).parent / "kits19" / "data"
PROCESSED_DIR = Path(__file__).parent / "kits19" / "processed"

"""
To download the dataset:
```sh
git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
cd ..
mv kits extra/datasets

Run to preprocess the datasets.
cd extra/datasets
PREPROCESS=1 python3 kits19.py
```
"""
@functools.lru_cache(None)
def get_val_files():
  data = fetch("https://raw.githubusercontent.com/mlcommons/training/master/image_segmentation/pytorch/evaluation_cases.txt").read_text()
  return sorted([x for x in BASEDIR.iterdir() if x.stem.split("_")[-1] in data.split("\n")])

def get_data_split(path: str="extra/datasets/kits19/processed", num_shards: int=0, shard_id: int=0):
  def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data
  def get_val_files_stem():
    data = fetch("https://raw.githubusercontent.com/mlcommons/training/master/image_segmentation/pytorch/evaluation_cases.txt").read_text()
    return sorted(data.split("\n"))
  val_cases_list = get_val_files_stem()
  val_cases_list = [case.rstrip("\n") for case in val_cases_list]
  imgs = load_data(path, "*_x.npy")
  lbls = load_data(path, "*_y.npy")
  assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks."
  imgs_train, lbls_train, imgs_val, lbls_val = [], [], [], []
  for (case_img, case_lbl) in zip(imgs, lbls):
    if case_img.split("_")[-2] in val_cases_list:
      imgs_val.append(case_img)
      lbls_val.append(case_lbl)
    else:
      imgs_train.append(case_img)
      lbls_train.append(case_lbl)
  return imgs_train, lbls_train, imgs_val, lbls_val

def save_kits19(image, label, case: str, results_dir: str):
  image = image.astype(np.float32)
  label = label.astype(np.uint8)
  mean, std = np.round(np.mean(image, (1, 2, 3)), 2), np.round(np.std(image, (1, 2, 3)), 2)
  np.save(os.path.join(results_dir, f"{case}_x.npy"), image, allow_pickle=False)
  np.save(os.path.join(results_dir, f"{case}_y.npy"), label, allow_pickle=False)

def load_pair(file_path):
  image, label = nib.load(os.path.join(file_path, "imaging.nii.gz")), nib.load(os.path.join(file_path, "segmentation.nii.gz"))
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

def preprocess_save(results_dir="kits19/processed", data_dir="kits19/data", max_id=210, excluded_cases=[]):
  os.makedirs(results_dir, exist_ok=True)
  for case in tqdm(sorted([f for f in os.listdir(data_dir) if "case" in f])):
    case_id = int(case.split("_")[1])
    if case_id in excluded_cases or case_id >= max_id:
      print("Case {}. Skipped.".format(case_id))
      continue
    image, label = preprocess(os.path.join(data_dir, case))
    save_kits19(image, label, case, results_dir)

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

def randrange(max_range):
  return 0 if max_range == 0 else random.randrange(max_range)

def get_cords(cord, idx, patch_size):
  return cord[idx], cord[idx] + patch_size[idx]

def _rand_crop(image, label, patch_size):
  ranges = [s - p for s, p in zip(image.shape[1:], patch_size)]
  cord = [randrange(x) for x in ranges]
  low_x, high_x = get_cords(cord, 0, patch_size)
  low_y, high_y = get_cords(cord, 1, patch_size)
  low_z, high_z = get_cords(cord, 2, patch_size)
  image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
  label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
  return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]

def _rand_foreg_cropd(image, label, patch_size):
  def adjust(foreg_slice, patch_size, label, idx):
    diff = patch_size[idx - 1] - (foreg_slice[idx].stop - foreg_slice[idx].start)
    sign = -1 if diff < 0 else 1
    diff = abs(diff)
    ladj = randrange(diff)
    hadj = diff - ladj
    low = max(0, foreg_slice[idx].start - sign * ladj)
    high = min(label.shape[idx], foreg_slice[idx].stop + sign * hadj)
    diff = patch_size[idx - 1] - (high - low)
    if diff > 0 and low == 0:
      high += diff
    elif diff > 0:
      low -= diff
    return low, high

  cl = np.random.choice(np.unique(label[label > 0]))
  foreg_slices = ndimage.find_objects(ndimage.measurements.label(label==cl)[0])
  foreg_slices = [x for x in foreg_slices if x is not None]
  slice_volumes = [np.prod([s.stop - s.start for s in sl]) for sl in foreg_slices]
  slice_idx = np.argsort(slice_volumes)[-2:]
  foreg_slices = [foreg_slices[i] for i in slice_idx]
  if not foreg_slices:
      return _rand_crop(image, label, patch_size)
  foreg_slice = foreg_slices[random.randrange(len(foreg_slices))]
  low_x, high_x = adjust(foreg_slice, patch_size, label, 1)
  low_y, high_y = adjust(foreg_slice, patch_size, label, 2)
  low_z, high_z = adjust(foreg_slice, patch_size, label, 3)
  image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
  label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
  return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]

def rand_crop(image, label, patch_size, oversampling):
  if random.random() < oversampling:
    image, label, cords = _rand_foreg_cropd(image, label, patch_size)
  else:
    image, label, cords = _rand_crop(image, label, patch_size)
  return image, label

def rand_flip(image, label, axis=[1,2,3]):
  prob = 1/len(axis)
  def _flip(image, label, axis):
    return np.flip(image, axis=axis).copy(), np.flip(label, axis=axis).copy()
  for ax in axis:
    if random.random() < prob:
      image, label = _flip(image, label, ax)
  return image, label

def cast(image, label, types=(np.float32, np.uint8)):
  return image.astype(types[0]), label.astype(types[1])

def rand_brightness(image, label, factor=0.3, prob=0.1):
  if random.random() < prob:
    fac = np.random.uniform(low=1.0-factor, high=1.0+factor, size=1)
    image = (image * (1 + fac)).astype(image.dtype)
  return image, label

def gaussian_noise(image, label, mean=0.0, std=0.1, prob=0.1):
  if random.random() < prob:
    scale = np.random.uniform(low=0.0, high=std)
    noise = np.random.normal(loc=mean, scale=scale, size=image.shape).astype(image.dtype)
    image += noise
  return image, label

def transform(X, Y, patch_size=(128,128,128), oversampling=0.25):
  X,Y = rand_flip(X,Y)
  X,Y = rand_brightness(X,Y)
  X,Y = gaussian_noise(X,Y)
  X,Y = rand_crop(X, Y, patch_size, oversampling)
  return X,Y

def get_batch(lX, lY, batch_size=32, patch_size=(128, 128, 128), oversampling=0.25, shuffle=True, augment=True):
  order = list(range(0, len(lX)))
  if shuffle: random.shuffle(order)
  for idxs in zip(*[iter(order)]* batch_size):
    bX, bY = [], []
    for i in idxs:
      X, Y = np.load(lX[i]), np.load(lY[i])
      if augment:
        X,Y = transform(X,Y, patch_size, oversampling)
      bX.append(X)
      bY.append(Y)
    yield (np.stack(bX, axis=0), np.stack(bY, axis=0))

if __name__ == "__main__":
  if getenv("PREPROCESS"): preprocess_save(); exit()

  for X, Y in iterate():
    print(X.shape, Y.shape)
