import random
from tinygrad.tensor import Tensor

import numpy as np
from torchvision.transforms import functional as F

class Compose(object):
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, image, target=None):
    print("compose")
    for t in self.transforms:
      if target:
        image, target = t(image, target)
        return image, target
      else:
        image = t(image)
        return image

  def __repr__(self):
    format_string = self.__class__.__name__ + "("
    for t in self.transforms:
      format_string += "\n"
      format_string += "    {0}".format(t)
    format_string += "\n)"
    return format_string

# TODO remove the dependence from torch here
class ToTensor(object):
  def __call__(self, image, target):
    print("Tensor")
    return F.to_tensor(image), target

class Resize:
  # Note
  # torch gives different output depends on whether the input images are PIL or
  # torch.tensor
  def __init__(self, min_size, max_size):
    if not isinstance(min_size, (list, tuple)):
      min_size = (min_size,)
    self.min_size = min_size
    self.max_size = max_size

  # modified from torchvision to add support for max size
  def get_size(self, image_size):
    w, h = image_size
    size = random.choice(self.min_size)
    max_size = self.max_size
    if max_size is not None:
      min_original_size = float(min((w, h)))
      max_original_size = float(max((w, h)))
      if max_original_size / min_original_size * size > max_size:
        size = int(round(max_size * min_original_size / max_original_size))

      if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

      if w < h:
        ow = size
        oh = int(size * h / w)
      else:
        oh = size
        ow = int(size * w / h)

      return (oh, ow)

  def __call__(self, image, target=None):
    print("resize")
    size = self.get_size(image.size)
    image = F.resize(image, size)
    if target:
      target = target.resize(image.size)
      return image, target
    else:
      return image

class RandomHorizontalFlip(object):
  def __init__(self, prob=0.5):
    self.prob = prob
  
  # TODO remove dependence from torchvision
  def __call__(self, image, target):

    if random.random() < self.prob:
      image = F.hflip(image)
      target = target.transpose(0)
    return image, target

class Normalize:
  def __init__(self, mean, std, to_bgr255=True):
    self.mean = mean
    self.std = std
    self.to_bgr255 = to_bgr255

  def __call__(self, image, target=None):
    # TODO the normalization mean and std are used from the inference 
    # implementation which appears to be RGB w/o scaling to 255
    # need to double check otherwise the entire training won't work.
    # Here are a informative explanation for the history of using BGR 
    # https://stackoverflow.com/questions/70115749/is-there-any-reason-for-changing-the-channels-order-of-an-image-from-rgb-to-bgr
    #
    # Values to be used for image normalization
    # `_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
    # ` Values to be used for image normalization
    # `_C.INPUT.PIXEL_STD = [1., 1., 1.]
    # ` Convert image to BGR format (for Caffe2 models), in range 0-255
    # `_C.INPUT.TO_BGR255 = True
    print("normalize")
    if self.to_bgr255:
      image = image[[2, 1, 0]] * 255
    image = Tensor(F.normalize(image, mean=self.mean, std=self.std).numpy())
    if target:
      return image, target
    else:
      print(image)
      return image