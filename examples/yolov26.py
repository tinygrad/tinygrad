import sys
from pathlib import Path

from tinygrad.helpers import fetch
from tinygrad import Tensor

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
    pass

class Upclass:
  pass

class Conv2d:
  pass

class BatchNorm2d:
  pass

class SiLU:
  pass

class Conv:
  def __init__(self, ch_in, ch_out, k, s) -> None:
    Conv2d()
    BatchNorm2d()
    SiLU()
    

if __name__=="__main__":
  print(f"sys.args: {sys.argv}")
  if len(sys.argv)<2:
    print("Error: image path is required")
    sys.exit(1)
  img_path = sys.argv[1]
  yolo_variant = sys.argv[2] if len(sys.argv)>=3 else (print("No variant given, so choosing 'n' as the default. Yolov8 has different variants, you can choose from ['n', 's', 'm', 'l', 'x']") or 'n') # default to nano
  print(f"Running inference for YOLO version {yolo_variant}")
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
