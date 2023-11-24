import re
import string
from collections import Counter
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes

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

def one_hot_np(arr: np.array, num_classes=3):
  res = np.eye(num_classes)[np.array(arr).reshape(-1)]
  arr = res.reshape(list(arr.shape) + [num_classes])
  arr = arr.transpose((0, 4, 1, 2, 3)).astype(np.float32)
  return arr

def one_hot(arr: Tensor, layout="NCDHW", channel_axis=1, num_classes=3):
  arr = arr.squeeze(dim=channel_axis)
  arr= Tensor.eye(num_classes, dtype=dtypes.int32, device=arr.device)[arr]
  arr= arr.permute(0, 4, 1, 2, 3)
  return arr

def get_dice_score_np(prediction, target, channel_axis=1, smooth_nr=1e-6, smooth_dr=1e-6):
  channel_axis, reduce_axis = 1, tuple(range(2, len(prediction.shape)))
  prediction = prediction.argmax(axis=channel_axis)
  prediction, target= one_hot_np(prediction)[:, 1:], one_hot_np(target)[:, 1:]
  intersection = np.sum(prediction * target, axis=reduce_axis)
  target_sum = np.sum(target, axis=reduce_axis)
  prediction_sum = np.sum(prediction, axis=reduce_axis)
  result = (2.0 * intersection + smooth_nr) / (target_sum + prediction_sum + smooth_dr)
  return result[0]

def get_dice_score(prediction, target, channel_axis=1, prediction_argmax=True, smooth_nr=1e-6, smooth_dr=1e-6):
  prediction, target = prediction.float(), target.float()
  reduce_axis = list(range(2, len(prediction.shape)))
  if prediction_argmax:
    assert not prediction.requires_grad
    prediction = prediction.argmax(axis=channel_axis)
    prediction = one_hot(prediction, channel_axis=channel_axis)
  else:
    prediction = prediction.softmax(axis=channel_axis)
  target = one_hot(target, channel_axis=channel_axis)

  prediction, target = prediction[:, 1:], target[:, 1:]
  assert target.shape == prediction.shape, f"Shapes do not match. prediction: ({prediction.shape}), target: ({target.shape})."
  intersection = (target * prediction).sum(axis=reduce_axis)
  target_sum = target.sum(axis=reduce_axis)
  prediction_sum = prediction.sum(axis=reduce_axis)
  return (2.0 * intersection + smooth_nr) / (target_sum + prediction_sum + smooth_dr)

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
