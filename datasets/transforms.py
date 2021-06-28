import numpy as np
from PIL import Image

from tinygrad.tensor import Tensor

class Compose:
  """Compose transforms
  """
  def __init__(self, trans):
    self.trans = trans

  def __call__(self, x):
    for t in self.trans:
      x = t(x)
    return x

class Resize:
  """Resize an image.
  """
  def __init__(self, shape):
    self.H, self.W = shape

  def __call__(self, x_in):
    def __im_resize(x): return x.resize((self.W, self.H))
    def __np_resize(x_in): 
      x = np.asarray(__im_resize(Image.fromarray(x_in.transpose(1, 2, 0), 'RGB'))).transpose(2, 0, 1).astype(np.float32) / 255.0
      return x if x_in.shape[0] != 1 else x[0:1]
    def __tinygrad_resize(x): return Tensor(__np_resize(x))
    if isinstance(x_in, np.ndarray): return __np_resize(x_in)
    elif isinstance(x_in, Tensor): return __tinygrad_resize(x)
    elif isinstance(x_in, Image.Image): return __im_resize(x)
    else: raise Exception(f'Error. Input type "{type(x)}" not supported for this transform.')

class G2RGB:
  """Convert a grayscale image into a rgb one, if needed.
  """
  def __call__(self, x): return np.tile(x, (3, 1, 1)) if x.shape[0] == 1 else x

class ToTensor:
  """Convert input into a TinyGrad Tensor.
  """
  def __call__(self, x):
    if isinstance(x, Image.Image): return Tensor((np.asarray(x) / 255.0).astype(np.float32)).transpose(order=(2, 0, 1))
    elif isinstance(x, np.ndarray): return Tensor(x)
    else: raise Exception(f'Error. Input type "{type(x)}" not supported for this transform.')

class Normalize:
  """Normalize the input given mean and std.
  """
  def __init__(self, mean, std):
    self.mean, self.std = np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)

  def __call__(self, x):
    mean = np.expand_dims(self.mean, axis=(1, 2))
    std = np.expand_dims(self.std, axis=(1, 2))
    x = (x - mean) / std
    return x
