import numpy as np
from scipy import signal

# ***** UNet3D *****

def gaussian_kernel(n, std):
  gaussian_1d = signal.gaussian(n, std)
  gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
  gaussian_3d = np.outer(gaussian_2d, gaussian_1d)
  gaussian_3d = gaussian_3d.reshape(n, n, n)
  gaussian_3d = np.cbrt(gaussian_3d)
  gaussian_3d /= gaussian_3d.max()
  return gaussian_3d


def prepare_arrays(image, roi_shape=(128, 128, 128)):
  assert len(roi_shape) == 3 and any(roi_shape)
  image_shape = list(image.shape[2:])
  result = np.zeros((1, 3, *image_shape), dtype=image.dtype)
  norm_map = np.zeros_like(result)
  norm_patch = gaussian_kernel(roi_shape[0], 0.125 * roi_shape[0]).astype(norm_map.dtype)
  return result, norm_map, norm_patch

def get_slice(image, roi_shape=(128, 128, 128), overlap_factor=0.5):
  assert len(roi_shape) == 3 and any(roi_shape)
  assert 0 < overlap_factor < 1
  image_shape, dim = list(image.shape[2:]), len(image.shape[2:])
  strides = [int(roi_shape[i] * (1 - overlap_factor)) for i in range(dim)]
  size = [(image_shape[i] - roi_shape[i]) // strides[i] + 1 for i in range(dim)]
  for i in range(0, strides[0] * size[0], strides[0]):
    for j in range(0, strides[1] * size[1], strides[1]):
      for k in range(0, strides[2] * size[2], strides[2]):
        yield i, j, k

def finalize(image, norm_map):
  image /= norm_map
  image = np.argmax(image, axis=1).astype(np.uint8)
  image = np.expand_dims(image, axis=0)
  return image

def one_hot(arr, channel_axis, num_classes=3):
  tmp = np.eye(num_classes)[np.array(arr).reshape(-1)]
  arr = tmp.reshape(list(arr.shape) + [num_classes])
  arr = arr.transpose((0, 4, 1, 2, 3)).astype(np.float64)
  return arr

def get_dice_score(case, prediction, target):
  channel_axis, reduce_axis, smooth_nr, smooth_dr = 1, (2, 3, 4), 1e-6, 1e-6
  prediction = one_hot(prediction, channel_axis)[:, 1:]
  target = one_hot(prediction, channel_axis)[:, :1]
  assert target.shape == prediction.shape
  intersection = np.sum(target * prediction, axis=reduce_axis)
  target_sum = np.sum(target, axis=reduce_axis)
  prediction_sum = np.sum(prediction, axis=reduce_axis)
  dice_val = (2.0 * intersection + smooth_nr) / (target_sum + prediction_sum + smooth_dr)
  return (case, dice_val[0])
