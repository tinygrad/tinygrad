import re
import string
from collections import Counter
import numpy as np

from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor


def levenshtein(a, b):
  n, m = len(a), len(b)
  if n > m:
    a, b, n, m = b, a, m, n

  current = list(range(n + 1))
  for i in range(1, m + 1):
    previous, current = current, [i] + [0] * n
    for j in range(1, n + 1):
      add, delete = previous[j] + 1, current[j - 1] + 1
      change = previous[j - 1]
      if a[j - 1] != b[i - 1]:
        change = change + 1
      current[j] = min(add, delete, change)

  return current[n]

def word_error_rate(x, y):
  scores = words = 0
  for h, r in zip(x, y):
    h_list = h.split()
    r_list = r.split()
    words += len(r_list)
    scores += levenshtein(h_list, r_list)
  return float(scores) / words, float(scores), words

def one_hot_np(arr, num_classes=3):
  res = np.eye(num_classes)[np.array(arr).reshape(-1)]
  arr = res.reshape(list(arr.shape) + [num_classes])
  arr = arr.transpose((0, 4, 1, 2, 3)).astype(np.float32)
  return arr

def to_one_hot(array: Tensor, layout="NCDHW", channel_axis=1):
  # https://github.com/mlcommons/training/blob/00f04c57d589721aabce4618922780d29f73cf4e/image_segmentation/pytorch/model/losses.py#L63
  array = array.squeeze(dim=channel_axis) #todo could move squeeze up
  num_classes = 3
  array = Tensor.eye(num_classes, dtype=dtypes.int32, device=array.device)[array] # this is the F.one_hot function
  array = array.permute(0, 4, 1, 2, 3)
  return array

def get_dice_score_np(prediction, target, channel_axis=1, smooth_nr=1e-6, smooth_dr=1e-6):
  channel_axis, reduce_axis = 1, tuple(range(2, len(prediction.shape)))
  prediction = prediction.argmax(axis=channel_axis)
  prediction, target= one_hot_np(prediction)[:, 1:], one_hot_np(target)[:, 1:]
  intersection = np.sum(prediction * target, axis=reduce_axis)
  target_sum = np.sum(target, axis=reduce_axis)
  prediction_sum = np.sum(prediction, axis=reduce_axis)
  result = (2.0 * intersection + smooth_nr) / (target_sum + prediction_sum + smooth_dr)
  return result[0]


def get_dice_score(prediction, target, prediction_argmax=True, smooth_nr=1e-6, smooth_dr=1e-6):
  prediction, target = prediction.float(), target.float()
  channel_axis = 1
  # both prediction and target should be one_hot
  # And only the prediction should be argmax
  reduce_axis = list(range(2, len(prediction.shape)))
  if prediction_argmax:
    assert not prediction.requires_grad
    prediction = prediction.argmax(axis=channel_axis)
    prediction = to_one_hot(prediction, channel_axis=channel_axis)
  else:
    prediction = prediction.softmax(axis=channel_axis)
  target = to_one_hot(target, channel_axis=channel_axis)

  target = target[:, 1:]
  prediction = prediction[:, 1:]

  assert target.shape == prediction.shape, f"Target and prediction shape do not match. Target: ({target.shape}), prediction: ({prediction.shape})."
  intersection = (target * prediction).sum(axis=reduce_axis)
  target_sum = target.sum(axis=reduce_axis)
  prediction_sum = prediction.sum(axis=reduce_axis)
  return (2.0 * intersection + smooth_nr) / (target_sum + prediction_sum + smooth_dr)

def dice_ce_loss(prediction, target):
  prediction, target = prediction.float(), target.float()
  # here prediction doesnt have one_hot. Only target has one_hot
  # here prediction has softmax - TRUE
  target_one_hot = to_one_hot(target) # we cant compute the dice_score with float16 because of overflows
  # todo use nn.cross entropy here?
  # overflow in reduce?????? Check with CPU=1
  # cross_entropy = -(target_one_hot * prediction.softmax(1).clip(1e-8, 1).log()).mean()
  cross_entropy = -(target_one_hot * prediction.softmax(1).clip(1e-8, 1).log()).mean()
  dice_score = get_dice_score(prediction, target, prediction_argmax=False)
  dice_loss = (1. - dice_score).mean()
  loss = (dice_loss + cross_entropy) / 2
  return loss

def normalize_string(s):
  s = "".join(c for c in s.lower() if c not in string.punctuation)
  s = re.sub(r'\b(a|an|the)\b', ' ', s)
  return " ".join(s.split())

def f1_score(x, y):
  xt = normalize_string(x).split()
  yt = normalize_string(y).split()
  ct = Counter(xt) & Counter(yt)
  if (ns := sum(ct.values())) == 0:
    return 0.0
  p = ns / len(xt)
  r = ns / len(yt)
  return 2 * p * r / (p + r)