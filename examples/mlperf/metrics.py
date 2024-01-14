import re
import string
from collections import Counter
from tinygrad import Tensor, dtypes

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

def one_hot(arr, num_classes=3, channel_axis=1):
  def _unshard_reshape_shard(x):
    from tinygrad.helpers import getenv
    if (gpus:=getenv("GPUS")) > 1:
      from tinygrad import Device
      x = x.to(Device.DEFAULT)
      x = x.reshape(-1)
      return x.shard_([f"GPU:{i}" for i in range(gpus)])
    return x

  if len(arr.shape) >= 5: arr = arr.squeeze(dim=channel_axis)
  arr_reshape = _unshard_reshape_shard(arr)
  res = Tensor.eye(num_classes)
  from tinygrad.helpers import getenv
  arr_reshape = arr
  if (gpus:=getenv("GPUS")) > 1:
    from tinygrad import Device
    arr_reshape = arr_reshape.to(Device.DEFAULT)
    arr_reshape = arr_reshape.reshape(-1)
    arr_reshape.shard_([f"GPU:{i}" for i in range(gpus)])
    res.shard_([f"GPU:{i}" for i in range(gpus)])
  res = res[arr_reshape]
  arr = res.reshape(list(arr.shape) + [num_classes])
  arr = arr.permute((0, 4, 1, 2, 3)).cast(dtypes.float)
  return arr

def dice_score(prediction, target, channel_axis=1, smooth_nr=1e-6, smooth_dr=1e-6, argmax=True, to_one_hot_x=True):
  channel_axis, reduce_axis = 1, tuple(range(2, len(prediction.shape)))
  if argmax: prediction = prediction.argmax(axis=channel_axis)
  else: prediction = prediction.softmax(axis=channel_axis)
  if to_one_hot_x: prediction = one_hot(prediction, channel_axis=channel_axis)
  target = one_hot(target, channel_axis=channel_axis)
  prediction, target = prediction[:, 1:], target[:, 1:]
  assert prediction.shape == target.shape, f"prediction ({prediction.shape}) and target ({target.shape}) shapes do not match"
  intersection = (prediction * target).sum(axis=reduce_axis)
  target_sum = target.sum(axis=reduce_axis)
  prediction_sum = prediction.sum(axis=reduce_axis)
  result = (2.0 * intersection + smooth_nr) / (target_sum + prediction_sum + smooth_dr)
  from tinygrad.helpers import getenv
  if (gpus:=getenv("GPUS")) > 1:
    from tinygrad import Device
    result = result.to(Device.DEFAULT)
    result = result[0]
    result.shard_([f"GPU:{i}" for i in range(gpus)])
  return result[0]

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
