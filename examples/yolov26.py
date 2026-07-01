import sys
from pathlib import Path

from tinygrad import Tensor
from tinygrad.helpers import fetch, make_tuple
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

def preprocess(image: np.ndarray, img_size=640, model_stride=32, model_pt=True) -> Tensor:
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

class ModuleList:
  def __init__(self,*layers):
    self.layers = layers
  def __iter__(self):
    return iter(self.layers)

class Sequential:
  def __init__(self, *layers):
    self.layers = layers
  def __call__(self,x:Tensor)->Tensor:
    for layer in self.layers:
      x=layer(x)
    return x

def autopad(k:int|tuple[int, ...], p:int|tuple[int, ...]|None=None, d:int=1):
  if d > 1:
    k = d * (k - 1) + 1 if isinstance(k, int) else tuple(d * (x - 1) + 1 for x in k)
  return (k // 2 if isinstance(k, int) else tuple(x // 2 for x in k)) if p is None else p

def depth_scale(n:int, d:float): return max(round(n * d), 1) if n > 1 else n

class YOLOv26:
  def __init__(self, w: float, d: float, ch: int, num_classes:int):
    self.backbone = Backbone(w,d,ch)
  def __call__(self, x: Tensor)->Tensor:
    return self.backbone(x)

class Upclass:
  pass

class Conv:
  def __init__(self, ch_in:int, ch_out:int, kernel_size:int|tuple[int,...], strides:int=1, padding:tuple|int|None=None, dilation:int=1, bias:bool=True, groups:int=1, act:bool=True) -> None:
    self.conv = Conv2d(ch_in, ch_out, kernel_size, strides, autopad(kernel_size, padding, dilation), dilation=dilation, groups=groups, bias=bias)
    self.bn = BatchNorm2d(ch_out)
    self.act = act
  
  def __call__(self, x:Tensor)->Tensor:
    x = self.bn(self.conv(x))
    return x.silu() if self.act else x

class Bottleneck:
  def __init__(self, c1, c2 , shortcut: bool, g=1, kernels:tuple = (3,3), channel_factor=0.5):
    c_ = int(c2 * channel_factor)
    self.cv1 = Conv(c1, c_, kernel_size=kernels[0], strides=1,)
    self.cv2 = Conv(c_, c2, kernel_size=kernels[1], strides=1, groups=g)
    self.residual = c1 == c2 and shortcut

  def __call__(self, x):
    return x + self.cv2(self.cv1(x)) if self.residual else self.cv2(self.cv1(x))

class C3K:
  def __init__(self,c1:int,c2:int,n:int=1,shortcut:bool=True,g:int=1,e:float=0.5,k:int|tuple[int, int]=3):
    c_=int(c2*e)
    k=make_tuple(k,2) if isinstance(k,int) else k
    self.cv1=Conv(c1,c_,1,1)
    self.cv2=Conv(c1,c_,1,1)
    self.cv3=Conv(2*c_,c2,1,1)
    self.m = Sequential(
      *(Bottleneck(c_,c_,shortcut,g,kernels=(k,k),channel_factor=1.0) for _ in range(n))
    )
  def __call__(self,x:Tensor)->Tensor:
    return self.cv3(self.m(self.cv1(x)).cat(self.cv2(x), dim=1))

class C2F:
  def __init__(self,c1:int,c2:int,n:int=1,shortcut:bool=False,g:int=1,e:float=0.5):
    self.c=int(c2*e)
    self.cv1=Conv(c1,2*self.c,1,1)
    self.cv2=Conv((2+n)*self.c,c2,1,1)
    self.m=ModuleList(*(Bottleneck(self.c,self.c,shortcut,g,kernels=((3,3),(3,3)),channel_factor=1.0) for _ in range(n)))
  def __call__(self,x:Tensor)->Tensor:
    y=list(self.cv1(x).chunk(2,1))
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(y[0].cat(*y[1:], dim=1))

class C3K2(C2F):
  def __init__(self,c1:int,c2:int,n:int=1,c3k:bool=False,e:float=0.5,g:int=1,shortcut:bool=True,k:int|tuple=3):
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
    super().__init__(c1,c2,n,shortcut,g,e)
    k=make_tuple(k,2) if isinstance(k,int) else k
    self.m=ModuleList(*(C3K(self.c,self.c,2,shortcut=shortcut,g=g,k=k) if c3k else Bottleneck(self.c,self.c,shortcut,g,kernels=(k,k)) for _ in range(n)))


class MaxPool2d:
  def __init__(self, kernel_size:int|tuple[int, int], stride:int|tuple[int, int]|None=None, padding:int|tuple[int, ...]=0, dilation:int|tuple[int, int]=1):
    self.kernel_size, self.stride, self.padding, self.dilation = kernel_size, stride, padding, dilation

  def __call__(self, x:Tensor)->Tensor:
    return x.max_pool2d(self.kernel_size, self.stride, self.dilation, self.padding)

class SPPF:
  def __init__(self, c1:int, c2:int, k:int=5, n:int=3, shortcut:bool=True):
    c_ = c1 // 2
    self.cv1 = Conv(c1, c_, 1, 1)
    self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
    self.m = MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    self.n = n
    self.residual = shortcut and c1 == c2

  def __call__(self, x:Tensor)->Tensor:
    y = [self.cv1(x)]
    y.extend(self.m(y[-1]) for _ in range(self.n))
    y = self.cv2(y[0].cat(*y[1:], dim=1))
    return x + y if self.residual else y

class Attention:
  def __init__(self, dim:int, num_heads:int=8, attn_ratio:float=0.5):
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.key_dim = int(self.head_dim * attn_ratio)
    self.scale = self.key_dim ** -0.5
    qkv_channels = dim + self.key_dim * num_heads * 2
    self.qkv = Conv(dim, qkv_channels, 1, act=False)
    self.proj = Conv(dim, dim, 1, act=False)
    self.pe = Conv(dim, dim, 3, groups=dim, act=False)

  def __call__(self, x:Tensor)->Tensor:
    b, c, h, w = x.shape
    qkv = self.qkv(x).reshape(b, self.num_heads, self.key_dim * 2 + self.head_dim, h * w)
    q, k, v = qkv.split((self.key_dim, self.key_dim, self.head_dim), dim=2)
    attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(-1)
    x = (v @ attn.transpose(-2, -1)).reshape(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
    return self.proj(x)

class PSABlock:
  def __init__(self, c:int, attn_ratio:float=0.5, num_heads:int=4, shortcut:bool=True):
    self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
    self.ffn = Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
    self.residual = shortcut

  def __call__(self, x:Tensor)->Tensor:
    x = x + self.attn(x) if self.residual else self.attn(x)
    return x + self.ffn(x) if self.residual else self.ffn(x)

class PSA:
  def __init__(self, c1:int, c2:int|None=None, e:float=0.5):
    c2 = c1 if c2 is None else c2
    assert c1 == c2
    self.c = int(c1 * e)
    self.cv1 = Conv(c1, 2 * self.c, 1, 1)
    self.cv2 = Conv(2 * self.c, c1, 1, 1)
    self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
    self.ffn = Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

  def __call__(self, x:Tensor)->Tensor:
    a, b = self.cv1(x).chunk(2, 1)
    b = b + self.attn(b)
    b = b + self.ffn(b)
    return self.cv2(a.cat(b, dim=1))

class C2PSA:
  def __init__(self, c1, c2, n=1, e=0.5):
    assert c1 == c2
    self.c = int(c1 * e)
    self.cv1 = Conv(c1, 2 * self.c, 1, 1)
    self.cv2 = Conv(2 * self.c, c1, 1, 1)
    self.m = Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n)))

  def __call__(self, x):
    a, b = self.cv1(x).chunk(2, 1)
    b = self.m(b)
    return self.cv2(a.cat(b, dim=1))

class Backbone:
  def __init__(self, w: float, d: float, ch: int):
    self.seq = Sequential(
      Conv(ch_in=3,ch_out=int(64*w),kernel_size=3,strides=2), # 0-P1/2 [-1, 1, Conv, [64, 3, 2]]
      Conv(ch_in=int(64*w),ch_out=int(128*w),kernel_size=3,strides=2), # 1-P2/4 [-1, 1, Conv, [128, 3, 2]]
      C3K2(c1=int(128*w),c2=int(256*w),n=depth_scale(2,d),e=0.25,k=3), # [-1, 2, C3k2, [256, False, 0.25]]
      Conv(ch_in=int(256*w),ch_out=int(256*w),kernel_size=3,strides=2), # 3-P3/8 [-1, 1, Conv, [256, 3, 2]]
      C3K2(c1=int(256*w),c2=int(512*w),n=depth_scale(2,d),e=0.25,k=3), # [-1, 2, C3k2, [512, False, 0.25]]
      Conv(ch_in=int(512*w),ch_out=int(512*w),kernel_size=3,strides=2), # 5-P4/16 [-1, 1, Conv, [512, 3, 2]] 
      C3K2(c1=int(512*w),c2=int(512*w),c3k=True,n=depth_scale(2,d),k=3), # [-1, 2, C3k2, [512, True]]
      Conv(ch_in=int(512*w),ch_out=int(1024*w),kernel_size=3,strides=2), # 7-P5/32 [-1, 1, Conv, [1024, 3, 2]]
      C3K2(c1=int(1024*w),c2=int(1024*w),c3k=True,n=depth_scale(2,d),k=3), # [-1, 2, C3k2, [1024, True]]
      SPPF(c1=int(1024*w),c2=int(1024*w),k=5, n=3, shortcut=True), # [-1, 1, SPPF, [1024, 5, 3, True]] # 9
      C2PSA(c1=int(1024*w),c2=int(1024*w),n=depth_scale(2,d)) # [-1, 2, C2PSA, [1024]] # 10
    )
  def __call__(self, x:Tensor):
    return self.seq(x)

if __name__=="__main__":
  print(f"sys.args: {sys.argv}")
  if len(sys.argv)<2:
    print("Error: image path is required")
    sys.exit(1)
  img_path = sys.argv[1]
  yolo_variant = sys.argv[2] if len(sys.argv)>=3 else (print("No variant given, so choosing 'n' as the default. Yolov8 has different variants, you can choose from ['n', 's', 'm', 'l', 'x']") or 'n') # default to nano
  print(f"args {sys.argv}")
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
  print(yolo_infer(preprocessed_image))
  sys.exit(0)
