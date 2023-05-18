# for kits19 clone the offical repo and run get_imaging.py
import random
import functools
from pathlib import Path
import pickle
import requests
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

BASEDIR = Path(__file__).parent.parent.parent.resolve() / "kits19" / "data"

@functools.lru_cache(None)
def get_val_files():
  eval_set = requests.get("https://raw.githubusercontent.com/mlcommons/inference/c04de44104b80b7950e027e6c4b9b025eff2338b/vision/medical_imaging/3d-unet-kits19/meta/inference_cases.json").json()
  return [x for x in BASEDIR.iterdir() if x.stem in eval_set]

# inspired by https://github.com/mlcommons/inference/blob/master/vision/medical_imaging/3d-unet-kits19/preprocess.py
def load_resampled_case(fn, target_spacing=[1.6, 1.2, 1.2]):
  image = nib.load(fn / "imaging.nii.gz")
  label = nib.load(fn / "segmentation.nii.gz")
  image_spacings = image.header["pixdim"][1:4].tolist()
  original_affine = image.affine
  image = image.get_fdata().astype(np.float32)
  label = label.get_fdata().astype(np.uint8)
  targ_arr = np.array(target_spacing)
  zoom_factor = np.array(target_spacing) / targ_arr
  reshaped_affine = original_affine.copy()
  for i in range(3):
    idx = np.where(original_affine[i][:-1] != 0)
    sign = -1 if original_affine[i][idx] < 0 else 1
    reshaped_affine[i][idx] = targ_arr[idx] * sign
    if image_spacings != target_spacing:
      image = zoom(image, zoom_factor, order=1, mode="constant", cval=image.min(), grid_mode=False)
      label = zoom(label, zoom_factor, order=0, mode="constant", cval=label.min(), grid_mode=False)
  aux = {"original_affine": original_affine, "reshaped_affine": reshaped_affine, "zoom_factor": zoom_factor, "case": fn.stem}
  image = np.expand_dims(image, 0)
  label = np.expand_dims(label, 0)
  return image, label, aux

def normalize_intensity(image, min_clip=-79.0, max_clip=304.0, mean=101.0, std=76.9):
  image = np.clip(image, min_clip, max_clip)
  image = (image - mean) / std
  return image

def pad_to_min_shape(image, label, roi_shape=[128, 128, 128]):
  current_shape = image.shape[1:]
  bounds = [max(0, roi_shape[i] - current_shape[i]) for i in range(3)]
  paddings = [(0, 0)] + [(bounds[i] // 2, bounds[i] - bounds[i] // 2) for i in range(3)]
  image = np.pad(image, paddings, mode="edge")
  label = np.pad(label, paddings, mode="edge")
  return image, label

def constant_pad_volume(volume, roi_shape, strides, padding_val, dim=3):
  bounds = [(strides[i] - volume.shape[1:][i] % strides[i]) % strides[i] for i in range(dim)]
  bounds = [bounds[i] if (volume.shape[1:][i] + bounds[i]) >= roi_shape[i] else bounds[i] + strides[i] for i in range(dim)]
  paddings = [(0, 0)] + [(bounds[i] // 2, bounds[i] - bounds[i] // 2) for i in range(3)]
  padded_volume = np.pad(volume, paddings, mode="constant", constant_values=[padding_val])
  return padded_volume, paddings

def adjust_shape(image, label, roi_shape=[128, 128, 128], overlap=0.5, padding_val=-2.2):
  image_shape = list(image.shape[1:])
  dim = len(image_shape)
  strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]
  bounds = [image_shape[i] % strides[i] for i in range(dim)]
  bounds = [bounds[i] if bounds[i] < strides[i] // 2 else 0 for i in range(dim)]
  image = image[..., bounds[0]//2:image_shape[0]-(bounds[0]-bounds[0]//2), bounds[1]//2:image_shape[1]-(bounds[1]-bounds[1]//2), bounds[2]//2:image_shape[2]-(bounds[2]-bounds[2]//2)]
  label = label[..., bounds[0]//2:image_shape[0]-(bounds[0]-bounds[0]//2), bounds[1]//2:image_shape[1]-(bounds[1]-bounds[1]//2), bounds[2]//2:image_shape[2]-(bounds[2]-bounds[2]//2)]
  image, paddings = constant_pad_volume(image, roi_shape, strides, padding_val)
  label, paddings = constant_pad_volume(label, roi_shape, strides, 0)
  return image, label

def save(image, label, aux):
  case = aux["case"]
  reshaped_affine = aux["reshaped_affine"]
  image = image.astype(np.float32)
  label = label.astype(np.uint8)
  mean, std = np.round(np.mean(image, (1, 2, 3)), 2), np.round(np.std(image, (1, 2, 3)), 2)
  results_dir = BASEDIR.parent / "results" 
  results_dir.mkdir(exist_ok=True) 
  with (results_dir / f"{case:05}.pkl").open("wb") as f:
    pickle.dump([image, label], f)

def preprocess(fn):
  image, label, aux = load_resampled_case(fn)
  image = normalize_intensity(image.copy())
  image, label = pad_to_min_shape(image, label)
  image, label = adjust_shape(image, label)
  save(image, label, aux)
  return image, label

def iterate(val=True, shuffle=True):
  if not val: raise NotImplementedError
  files = get_val_files()
  order = list(range(0, len(files)))
  if shuffle: random.shuffle(order)
  for file in files:
    result_file = file.parent.parent / "results" / f"{file.stem}.pkl"
    if result_file.is_file():
      with result_file.open("rb") as f:
        X, Y = pickle.load(f)
    else:
      X, Y = preprocess(file)
    yield (np.array(X), np.array(Y), file.stem)

if __name__ == "__main__":
  X, Y = next(iterate())
  print(X.shape, Y.shape)
