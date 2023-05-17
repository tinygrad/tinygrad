import numpy as np

# ***** UNet3D *****

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
