import os
import argparse
import numpy as np
from skimage import io

from typing import Tuple

# TODO:
# Remove torch
#  Implement F.interpolate
#  Implement transforms.normalize
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

import tinygrad.nn as nn
from tinygrad import Tensor
from tinygrad.nn.state import load_state_dict
from tinygrad.helpers import fetch


def maxpool2d(x: Tensor, kernel_size=2, stride=2) -> Tensor:
  return x.max_pool2d(kernel_size=(kernel_size, kernel_size), stride=stride)


def _upsample_like(src: Tensor, tar: Tensor) -> Tensor:
  # TODO: Get rid of torch dependency here.
  src = torch.tensor(src.numpy())
  result = F.interpolate(src, size=tar.shape[2:], mode="bilinear")
  return Tensor(result.cpu().numpy())


# Adapted from https://github.com/xuebinqin/DIS.
class REBNCONV:
  def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
    self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1 * dirate, dilation=1 * dirate)
    self.bn_s1 = nn.BatchNorm2d(out_ch)

  def __call__(self, x: Tensor) -> Tensor:
    return self.bn_s1(self.conv_s1(x)).relu()


class RSU7:
  def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
    self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
    self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
    self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
    self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
    self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
    self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
    self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
    self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

    self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
    self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
    self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
    self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
    self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
    self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

  def __call__(self, x: Tensor) -> Tensor:
    hx = x
    hxin = self.rebnconvin(hx)

    hx1 = self.rebnconv1(hxin)
    hx = maxpool2d(hx1)

    hx2 = self.rebnconv2(hx)
    hx = maxpool2d(hx2)

    hx3 = self.rebnconv3(hx)
    hx = maxpool2d(hx3)

    hx4 = self.rebnconv4(hx)
    hx = maxpool2d(hx4)

    hx5 = self.rebnconv5(hx)
    hx = maxpool2d(hx5)

    hx6 = self.rebnconv6(hx)
    hx7 = self.rebnconv7(hx6)

    hx6d = self.rebnconv6d(hx7.cat(hx6, dim=1))
    hx6dup = _upsample_like(hx6d, hx5)

    hx5d = self.rebnconv5d(hx6dup.cat(hx5, dim=1))
    hx5dup = _upsample_like(hx5d, hx4)

    hx4d = self.rebnconv4d(hx5dup.cat(hx4, dim=1))
    hx4dup = _upsample_like(hx4d, hx3)

    hx3d = self.rebnconv3d(hx4dup.cat(hx3, dim=1))
    hx3dup = _upsample_like(hx3d, hx2)

    hx2d = self.rebnconv2d(hx3dup.cat(hx2, dim=1))
    hx2dup = _upsample_like(hx2d, hx1)

    hx1d = self.rebnconv1d(hx2dup.cat(hx1, dim=1))

    return hx1d + hxin


### RSU-6 ###
class RSU6:
  def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
    self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
    self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
    self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
    self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
    self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
    self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
    self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

    self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
    self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
    self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
    self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
    self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

  def __call__(self, x):
    hx = x

    hxin = self.rebnconvin(hx)

    hx1 = self.rebnconv1(hxin)
    hx = maxpool2d(hx1)

    hx2 = self.rebnconv2(hx)
    hx = maxpool2d(hx2)

    hx3 = self.rebnconv3(hx)
    hx = maxpool2d(hx3)

    hx4 = self.rebnconv4(hx)
    hx = maxpool2d(hx4)

    hx5 = self.rebnconv5(hx)

    hx6 = self.rebnconv6(hx5)

    hx5d = self.rebnconv5d(hx6.cat(hx5, dim=1))
    hx5dup = _upsample_like(hx5d, hx4)

    hx4d = self.rebnconv4d(hx5dup.cat(hx4, dim=1))
    hx4dup = _upsample_like(hx4d, hx3)

    hx3d = self.rebnconv3d(hx4dup.cat(hx3, dim=1))
    hx3dup = _upsample_like(hx3d, hx2)

    hx2d = self.rebnconv2d(hx3dup.cat(hx2, dim=1))
    hx2dup = _upsample_like(hx2d, hx1)

    hx1d = self.rebnconv1d(hx2dup.cat(hx1, dim=1))

    return hx1d + hxin


### RSU-5 ###
class RSU5:
  def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
    self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
    self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
    self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
    self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
    self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
    self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

    self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
    self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
    self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
    self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

  def __call__(self, x):
    hx = x

    hxin = self.rebnconvin(hx)

    hx1 = self.rebnconv1(hxin)
    hx = maxpool2d(hx1)

    hx2 = self.rebnconv2(hx)
    hx = maxpool2d(hx2)

    hx3 = self.rebnconv3(hx)
    hx = maxpool2d(hx3)

    hx4 = self.rebnconv4(hx)

    hx5 = self.rebnconv5(hx4)

    hx4d = self.rebnconv4d(hx5.cat(hx4, dim=1))
    hx4dup = _upsample_like(hx4d, hx3)

    hx3d = self.rebnconv3d(hx4dup.cat(hx3, dim=1))
    hx3dup = _upsample_like(hx3d, hx2)

    hx2d = self.rebnconv2d(hx3dup.cat(hx2, dim=1))
    hx2dup = _upsample_like(hx2d, hx1)

    hx1d = self.rebnconv1d(hx2dup.cat(hx1, dim=1))

    return hx1d + hxin


### RSU-4 ###
class RSU4:
  def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
    self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
    self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
    self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
    self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
    self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

    self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
    self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
    self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

  def __call__(self, x):
    hx = x

    hxin = self.rebnconvin(hx)

    hx1 = self.rebnconv1(hxin)
    hx = maxpool2d(hx1)

    hx2 = self.rebnconv2(hx)
    hx = maxpool2d(hx2)

    hx3 = self.rebnconv3(hx)

    hx4 = self.rebnconv4(hx3)

    hx3d = self.rebnconv3d(hx4.cat(hx3, dim=1))
    hx3dup = _upsample_like(hx3d, hx2)

    hx2d = self.rebnconv2d(hx3dup.cat(hx2, dim=1))
    hx2dup = _upsample_like(hx2d, hx1)

    hx1d = self.rebnconv1d(hx2dup.cat(hx1, dim=1))

    return hx1d + hxin


### RSU-4F ###
class RSU4F:
  def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
    self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

    self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
    self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
    self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

    self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

    self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
    self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
    self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

  def __call__(self, x):
    hx = x

    hxin = self.rebnconvin(hx)

    hx1 = self.rebnconv1(hxin)
    hx2 = self.rebnconv2(hx1)
    hx3 = self.rebnconv3(hx2)

    hx4 = self.rebnconv4(hx3)

    hx3d = self.rebnconv3d(hx4.cat(hx3, dim=1))
    hx2d = self.rebnconv2d(hx3d.cat(hx2, dim=1))
    hx1d = self.rebnconv1d(hx2d.cat(hx1, dim=1))

    return hx1d + hxin


class ISNetDIS:
  def __init__(self, in_ch=3, out_ch=1):
    self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
    self.stage1 = RSU7(64, 32, 64)
    self.stage2 = RSU6(64, 32, 128)
    self.stage3 = RSU5(128, 64, 256)
    self.stage4 = RSU4(256, 128, 512)
    self.stage5 = RSU4F(512, 256, 512)
    self.stage6 = RSU4F(512, 256, 512)

    # decoder
    self.stage5d = RSU4F(1024, 256, 512)
    self.stage4d = RSU4(1024, 128, 256)
    self.stage3d = RSU5(512, 64, 128)
    self.stage2d = RSU6(256, 32, 64)
    self.stage1d = RSU7(128, 16, 64)

    self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)

  def __call__(self, x):
    hx = x

    hxin = self.conv_in(hx)

    # stage 1
    hx1 = self.stage1(hxin)
    hx = maxpool2d(hx1)

    # stage 2
    hx2 = self.stage2(hx)
    hx = maxpool2d(hx2)

    # stage 3
    hx3 = self.stage3(hx)
    hx = maxpool2d(hx3)

    # stage 4
    hx4 = self.stage4(hx)
    hx = maxpool2d(hx4)

    # stage 5
    hx5 = self.stage5(hx)
    hx = maxpool2d(hx5)

    # stage 6
    hx6 = self.stage6(hx)
    hx6up = _upsample_like(hx6, hx5)

    # -------------------- decoder --------------------
    hx5d = self.stage5d(hx6up.cat(hx5, dim=1))
    hx5dup = _upsample_like(hx5d, hx4)

    hx4d = self.stage4d(hx5dup.cat(hx4, dim=1))
    hx4dup = _upsample_like(hx4d, hx3)

    hx3d = self.stage3d(hx4dup.cat(hx3, dim=1))
    hx3dup = _upsample_like(hx3d, hx2)

    hx2d = self.stage2d(hx3dup.cat(hx2, dim=1))
    hx2dup = _upsample_like(hx2d, hx1)

    hx1d = self.stage1d(hx2dup.cat(hx1, dim=1))

    d1 = self.side1(hx1d)
    d1 = _upsample_like(d1, x)
    return d1.sigmoid()


def load_model(model_path) -> ISNetDIS:
  net = ISNetDIS()
  weights = nn.state.torch_load(model_path)
  for key in list(weights.keys()):
    if key.endswith("num_batches_tracked"):
      del weights[key]
  load_state_dict(net, weights, strict=False)
  return net


def load_and_preprocess_image(im_path) -> Tuple[Tensor, np.array]:
  im = io.imread(im_path)
  if len(im.shape) < 3:
    im = im[:, :, np.newaxis]
  if im.shape[2] > 3:
    # We should probably blend pre-existing alpha instead?
    im = im[:, :, :3]
  im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
  im_tensor = F.interpolate(im_tensor, [1024, 1024], mode="bilinear").type(torch.uint8).div(255.0)
  im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
  return Tensor(im_tensor.numpy()), im


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Remove background from image",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--input", type=str, default="", help="Path to image")
  parser.add_argument(
    "--output",
    type=str,
    default="",
    help="Path to output image (removed background)",
  )
  args = parser.parse_args()

  im_tensor, original = load_and_preprocess_image(args.input)

  # Or download the original one from https://github.com/xuebinqin/DIS.
  net = load_model(fetch("https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth"))

  result = net(im_tensor)

  # Upsample result back to original size.
  result = torch.tensor(result.numpy())
  result = F.interpolate(result, original.shape[:2], mode="bilinear").squeeze(0)

  # Extract mask and apply as alpha channel to the image.
  ma = torch.max(result)
  mi = torch.min(result)
  result = (result - mi) / (ma - mi)
  mask = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
  image = np.dstack((original, mask))

  out_path = args.output
  if not out_path:
    out_path = "./output.png"
  io.imsave(out_path, image)
  print(f"Saved result in {out_path}.")
