import os
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import trange

from extra.utils import download_file
from tinygrad.helpers import getenv, prod
from tinygrad.nn import Conv2d, ConvTranspose2d, InstanceNorm, optim
from tinygrad.state import get_parameters
from tinygrad.tensor import Tensor


### Architecture definition
# Based off https://github.com/mlcommons/training/blob/master/image_segmentation/pytorch/model/unet3d.py

class DownsampleBlock:
  def __init__(self, in_channels, out_channels, stride_in=2):
    # Conv2d is a hidden ConvNd, here we are using Conv3d in particular
    self.conv1 = Conv2d(in_channels, out_channels, kernel_size=(3,3,3), stride=stride_in, padding=1, bias=False)
    self.instnorm1 = InstanceNorm(out_channels)
    self.conv2 = Conv2d(out_channels, out_channels, kernel_size=(3,3,3), stride=1, padding=1, bias=False)
    self.instnorm2 = InstanceNorm(out_channels)

  def __call__(self, x):
    x = self.instnorm1(self.conv1(x)).relu()
    x = self.instnorm2(self.conv2(x)).relu()
    return x

class OutputLayer:
  def __init__(self, in_channels, n_class):
    self.conv = Conv2d(in_channels, n_class, kernel_size=(1,1,1), stride=1, padding=0, bias=True)

  def __call__(self, x):
    # The relu activation is not in the MLPerf implementation
    return self.conv(x).relu()

class UpsampleBlock:
  def __init__(self, in_channels, out_channels):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.upsample_conv = ConvTranspose2d(in_channels, out_channels, kernel_size=(2,2,2), stride=2, padding=(1,0,0), bias=False)
    self.conv1 = Conv2d(2 * out_channels, out_channels, kernel_size=(3,3,3), padding=1, bias=False)
    self.instnorm1 = InstanceNorm(out_channels)
    self.conv2 = Conv2d(out_channels, out_channels, kernel_size=(3,3,3), padding=1, bias=False)
    self.instnorm2 = InstanceNorm(out_channels)

  def __call__(self, x, skip):
    x = self.upsample_conv(x, skip.shape, num_spatial_dims=3)
    x = x.cat(skip, dim=1)
    x = self.instnorm1(self.conv1(x)).relu()
    x = self.instnorm2(self.conv2(x)).relu()
    return x

class Unet3D:
  def __init__(self, in_channels, n_class):
    filters = [32, 64, 128, 256, 320]
    self.filters = filters
    self.inp = filters[:-1]
    self.out = filters[1:]
    input_dim = filters[0]
    self.input_block = DownsampleBlock(in_channels, input_dim, stride_in=1)
    self.downsample =  [DownsampleBlock(i, o) for (i, o) in zip(self.inp, self.out)]
    self.bottleneck = DownsampleBlock(filters[-1], filters[-1])
    self.upsample = [UpsampleBlock(filters[-1], filters[-1])]
    self.upsample.extend([UpsampleBlock(i, o) for (i, o) in zip(reversed(self.out), reversed(self.inp))])
    self.output = OutputLayer(input_dim, n_class)

  def __call__(self, x):
    x = self.input_block(x)
    outputs = [x]
    for downsample in self.downsample:
      x = downsample(x)
      outputs.append(x)
    x = self.bottleneck(x)
    for upsample, skip in zip(self.upsample, reversed(outputs)):
      x = upsample(x, skip)
    x = self.output(x)
    return x


### Data download/loading ###
# Data is from the KITS19 challenge: https://kits19.grand-challenge.org/

def fetch_kits19(num_cases=210, train_test_split=0.8):
  # Images have different dimensions: (C, 512, 512), and segmentations have 3 classes (including background)
  if not Path("data").exists():
    Path("data").mkdir()
  X_train, Y_train, X_test, Y_test = [], [], [], []
  for i in range(num_cases):
    print("{}/{}... ".format(i+1, num_cases))
    imaging, segmentation = get_case(i)
    if i < int(num_cases * train_test_split):
      X_train.append(imaging)
      Y_train.append(segmentation)
    else:
      X_test.append(imaging)
      Y_test.append(segmentation)
  exit()
  return X_train, Y_train, X_test, Y_test

def get_case(cid):
  BASE = Path(__file__).parent.parent / "data"
  cid = f"{cid:05d}"
  imaging_url = f"https://kits19.sfo2.digitaloceanspaces.com/master_{cid}.nii.gz"
  imaging_fp = os.path.join(BASE, f"case_{cid}", "imaging.nii.gz")
  download_file(imaging_url, imaging_fp)
  segmentation_url = f"https://github.com/neheller/kits19/raw/master/data/case_{cid}/segmentation.nii.gz" # f"https://raw.githubusercontent.com/neheller/kits19/blob/master/data/case_{cid}/segmentation.nii.gz"
  segmentation_pf = os.path.join(BASE, f"case_{cid}", "segmentation.nii.gz")
  download_file(segmentation_url, segmentation_pf)
  # https://nipy.org/nibabel/images_and_memory.html#use-the-array-proxy-instead-of-get-fdata
  imaging = nib.load(imaging_fp).dataobj
  segmentation = nib.load(segmentation_pf).dataobj
  return imaging, segmentation


#### Dice loss

def dice(prediction, target, smooth_nr=1e-6, smooth_dr=1e-6):
  channel_axis = 1
  reduce_axis = list(range(2, len(prediction.shape)))
  prediction = prediction.softmax(channel_axis)
  assert target.shape == prediction.shape, f"Target and prediction shape do not match. Target: ({target.shape}), prediction: ({prediction.shape})."
  intersection = (target * prediction).sum(axis=reduce_axis)
  target_sum = target.sum(axis=reduce_axis)
  prediction_sum = prediction.sum(axis=reduce_axis)
  return (2.0 * intersection + smooth_nr) / (target_sum + prediction_sum + smooth_dr)

def dice_ce_loss(y_pred, y_true):
  cross_entropy = -y_true.mul(y_pred.clip(1e-10, 1).log()).mean()
  dice_score = dice(y_pred, y_true)
  dice_loss = (Tensor.ones_like(dice_score) - dice_score).mean()
  loss = (dice_loss + cross_entropy) / 2
  return loss, cross_entropy, dice_score.mean()


##### Training and evaluation

def get_one_hot(targets, nb_classes):
  res = np.eye(nb_classes)[targets.reshape(-1)]
  return res.reshape([1, nb_classes]+list(targets.shape)).astype(np.float32)

def epoch(is_training, model, X, Y, optim, n_classes, noloss=False):
  Tensor.training = is_training
  for step in (t := trange(len(X), disable=getenv('CI', False))):
    x = Tensor(np.asarray(X[step][:50,:50,:50], dtype=np.float32), requires_grad=False).unsqueeze(0).unsqueeze(0)
    y = Y[step]
    y = np.asarray(y, dtype=np.int32)[:50,:50,:50]
    y = get_one_hot(y, n_classes)
    y = Tensor(y, requires_grad=False)
    out = model(x)
    loss, ce, dice = dice_ce_loss(out, y)
    if is_training:
      optim.zero_grad()
      loss.backward()
      if noloss: del loss
      optim.step()
    if not noloss:
      if is_training:
        t.set_description(f"loss {loss.numpy():.5f} cross entropy {ce.numpy():.5f} dice {dice.numpy():.5f}")
      else:
        t.set_description(f"val loss {loss.numpy():.5f} val cross entropy {ce.numpy():.5f} val dice {dice.numpy():.5f}")


if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = fetch_kits19()
  n_channels, n_class = 1, 3
  model = Unet3D(n_channels, n_class)
  lr = 5e-3
  for _ in range(5):
    optimizer = optim.SGD(get_parameters(model), lr=lr, momentum=0.9)
    epoch(True, model, X_train, Y_train, optimizer, n_class)
    epoch(False, model, X_test, Y_test, None, n_class)
    lr /= 1.2
    print(f'reducing lr to {lr:.7f}')
