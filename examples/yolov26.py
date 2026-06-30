import sys
from pathlib import Path

from tinygrad import Tensor
from tinygrad.helpers import fetch
from tinygrad.nn import Conv2d, BatchNorm2d

import cv2
import numpy as np

def compute_transform(image:np.ndarray,new_shape=(640,640),auto=False,scaleFill=False,scaleUp=True,stride=32):
  shape = image.shape[:2] # current shape [height, width]
  new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape
  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) # get scale factor
  r = min(r, 1.0) if not scaleUp else r
  new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r))) # scale image
  dw,dh = new_shape[1]-new_unpad[1], new_shape[0]-new_unpad[0]
  dw,dh = (np.mod(dw, stride), np.mod(dh, stride)) if auto else (0.0, 0.0) # if auto: add enough padding so that strides divide both dims
  new_unpad = (new_shape[1], new_shape[0]) if scaleFill else new_unpad
  dw /= 2
  dh /= 2
  image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR) if shape[::-1] != new_unpad else image
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
  return Tensor(image)

def preprocess(image: np.ndarray, img_size=640, model_stride=32, model_pt=True):
  im = compute_transform(image, new_shape=img_size, auto=model_pt, stride=model_stride)
  im = im.unsqueeze(0)
  im = im[..., ::-1].permute(0,3,1,2) # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
  im = im / 255.0
  return im

def get_variant_scales(variant: str):
  """
  https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/26/yolo26.yaml
  [depth, width, max_channels]
  """
  VARIANTS = {
    "n": (0.5, 0.25, 1024),
    "s": (0.5, 0.5, 1024),
    "m": (0.5, 1.0, 512),
    "l": (1.0, 1.0, 512),
    "x": (1.0, 1.5, 512)
  }
  return VARIANTS[variant]

class YOLOv26:
  def __init__(self, w: float, h: float, ch: int, num_classes:int):
    self.backbone = Backbone(w,h,ch)
  def __call__(self, x: Tensor):
    pass

class Upclass:
  pass

class Conv:
  def __init__(self, ch_in:int, ch_out:int, kernel_size:int|tuple[int,...], strides:int, padding:tuple|int=0, dilation:int=1, bias:bool=True) -> None:
    self.conv = Conv2d(ch_in, ch_out, kernel_size, strides, padding, dilation, bias)
    self.bn = BatchNorm2d(ch_out)
  
  def __call__(self, x:Tensor)->Tensor:
    x = self.bn(self.conv(x)).silu()

class Bottleneck:
  def __init__(self, c1, c2 , shortcut: bool, g=1, kernels: list = (3,3), channel_factor=0.5):
    c_ = int(c2 * channel_factor)
    self.cv1 = Conv(c1, c_, kernel_size=kernels[0], stride=1, padding=None)
    self.cv2 = Conv(c_, c2, kernel_size=kernels[1], stride=1, padding=None, groups=g)
    self.residual = c1 == c2 and shortcut

  def __call__(self, x):
    return x + self.cv2(self.cv1(x)) if self.residual else self.cv2(self.cv1(x))

class C3K:
  def __init__(self,c1:int,c2:int,n:int=1,c3k:bool=False,e:float=0.5,g:int=1,shortcut:bool=True):
    pass

class C3K2:
  def __init__(self,c1:int,c2:int,n:int=1,c3k:bool=False,e:float=0.5,g:int=1,shortcut:bool=True):
    """
      Initialize C3k2 module.

      Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of blocks.
        c3k (bool): Whether to use C3k blocks.
        e (float): Expansion ratio.
        g (int): Groups for convolutions.
        shortcut (bool): Whether to use shortcut connections.
    """
    self.c = int(c2 * e)
    self.cv1 = Conv(c1, 2 * self.c, 1,)
    self.cv2 = Conv((2 + n) * self.c, c2, 1)
    self.c3k = [C3K(self.c, self.c, shortcut, g, kernels=[(3, 3), (3, 3)], channel_factor=1.0) for _ in range(n)]

class Sequential:
  pass

class Backbone:
  def __init__(self, w: float, h: float, ch: int):
    self.conv = Conv(ch_in=3,ch_out=int(64*w),kernel_size=3,strides=2) ## 0-P1/2 [-1, 1, Conv, [64, 3, 2]]
    self.conv = Conv(ch_in=int(64*w),ch_out=int(128*w),kernel_size=3,strides=2) # 1-P2/4 [-1, 1, Conv, [128, 3, 2]]
    self.c3k2
  def __call__(self, x:Tensor):
    pass

if __name__=="__main__":
  print(f"sys.args: {sys.argv}")
  if len(sys.argv)<2:
    print("Error: image path is required")
    sys.exit(1)
  img_path = sys.argv[1]
  yolo_variant = sys.argv[2] if len(sys.argv)>=3 else (print("No variant given, so choosing 'n' as the default. Yolov8 has different variants, you can choose from ['n', 's', 'm', 'l', 'x']") or 'n') # default to nano
  print(f"Running inference for YOLO26 variant {yolo_variant}")
  (output_folder_path := Path('./outputs-yolov26')).mkdir(parents=True, exist_ok=True)
  image_location = np.frombuffer(fetch(img_path).read_bytes(), dtype=np.uint8)
  image = cv2.imdecode(image_location, 1)
  out_path = (output_folder_path / f"{Path(img_path).stem}_output{Path(img_path).suffix or '.png'}").as_posix()
  if not isinstance(image, np.ndarray):
    print('Error in image loading. Check your image file.')
    sys.exit(1)
  preprocessed_image = preprocess(image)
  depth, width, max_channels = get_variant_scales(yolo_variant)
  yolo_infer = YOLOv26(w=width, d=depth, ch=max_channels, num_classes=80)
