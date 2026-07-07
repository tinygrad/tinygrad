import json
import sys
import cv2
import math
import copy
import numpy as np
from time import time
from itertools import chain
from pathlib import Path
from collections import defaultdict

from tinygrad import Tensor
from tinygrad.nn import Conv2d, BatchNorm2d
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.helpers import fetch, make_tuple

def compute_transform(image:np.ndarray,new_shape=(640,640),auto=False,scaleFill=False,scaleUp=True,stride=32):
  shape = image.shape[:2] # current shape [height, width]
  new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape
  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) # get scale factor
  r = min(r, 1.0) if not scaleUp else r
  new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r))) # scale image
  dw,dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
  dw,dh = (np.mod(dw, stride), np.mod(dh, stride)) if auto else (dw, dh) # if auto: add enough padding so that strides divide both dims
  if scaleFill:
    dw, dh = 0.0, 0.0
    new_unpad = (new_shape[1], new_shape[0])
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

def autopad(k:int|tuple[int, ...], p:int|tuple[int, ...]|None=None, d:int=1):
  if d > 1:
    k = d * (k - 1) + 1 if isinstance(k, int) else tuple(d * (x - 1) + 1 for x in k)
  return (k // 2 if isinstance(k, int) else tuple(x // 2 for x in k)) if p is None else p

def depth_scale(n:int, d:float): return max(round(n * d), 1) if n > 1 else n

class Conv:
  def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True) -> None:
    self.conv = Conv2d(c1, c2, k, s, autopad(k,p,d), dilation=d, groups=g, bias=False)
    self.bn = BatchNorm2d(c2, eps=0.001)
    self.act = act
  
  def __call__(self, x:Tensor)->Tensor:
    return self.bn(self.conv(x)).silu() if self.act else self.bn(self.conv(x))

class Bottleneck:
  def __init__(self, c1, c2 , shortcut: bool, g=1, kernels:tuple = (3,3), e:float=0.5):
    c_ = int(c2 * e)
    self.cv1 = Conv(c1, c_, k=kernels[0], s=1)
    self.cv2 = Conv(c_, c2, k=kernels[1], s=1, g=g)
    self.add = c1 == c2 and shortcut

  def __call__(self, x):
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3K:
  def __init__(self,c1:int,c2:int,n:int=1,shortcut:bool=True,g:int=1,e:float=0.5,k:int|tuple[int, int]=3):
    c_=int(c2*e)
    k=make_tuple(k,2) if isinstance(k,int) else k
    self.cv1=Conv(c1,c_,1,1)
    self.cv2=Conv(c1,c_,1,1)
    self.cv3=Conv(2*c_,c2,1,1)
    self.m = [Bottleneck(c_,c_,shortcut,g,kernels=(k,k),e=1.0) for _ in range(n)]
    
  def __call__(self,x:Tensor)->Tensor:
    y1 = self.cv1(x)
    for block in self.m:
      y1 = block(y1)
    y2 = self.cv2(x)
    return self.cv3(y1.cat(y2, dim=1))

class C2F:
  def __init__(self,c1:int,c2:int,n:int=1,shortcut:bool=False,g:int=1,e:float=0.5):
    self.c=int(c2*e)
    self.cv1=Conv(c1,2*self.c,1,1)
    self.cv2=Conv((2+n)*self.c,c2,1)
    self.m=[Bottleneck(self.c,self.c,shortcut,g,kernels=((3,3),(3,3)),e=1.0) for _ in range(n)]
  def __call__(self,x:Tensor)->Tensor:
    y=list(self.cv1(x).chunk(2,1))
    for block in self.m:
      y.append(block(y[-1]))
    return self.cv2(y[0].cat(*y[1:], dim=1))

class C3K2(C2F):
  def __init__(self,c1:int,c2:int,n:int=1,c3k:bool=False,e:float=0.5,g:int=1,shortcut:bool=True,k:int|tuple=3,attn=False):
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
    self.m = [
      [
        Bottleneck(self.c, self.c, shortcut, g),
        PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
      ]
      if attn else 
      C3K(self.c, self.c, 2, shortcut=shortcut, g=g, k=k)
      if c3k else
      Bottleneck(self.c, self.c, shortcut, g)
      for _ in range(n)
    ]

  def __call__(self, x: Tensor) -> Tensor:
    y = list(self.cv1(x).chunk(2, 1))
    for block in self.m:
      z = y[-1]
      if isinstance(block, list):
        for layer in block:
          z = layer(z)
      else:
        z = block(z)
      y.append(z)
    return self.cv2(y[0].cat(*y[1:], dim=1))

class MaxPool2d:
  def __init__(self, kernel_size:int|tuple[int, int], stride:int|tuple[int, int]|None=None, padding:int|tuple[int, ...]=0, dilation:int|tuple[int, int]=1):
    self.kernel_size, self.stride, self.padding, self.dilation = kernel_size, stride, padding, dilation

  def __call__(self, x:Tensor)->Tensor:
    return x.max_pool2d(self.kernel_size, self.stride, self.dilation, self.padding)

class SPPF:
  def __init__(self, c1:int, c2:int, k:int=5, n:int=3, shortcut:bool=True):
    c_ = c1 // 2
    self.cv1 = Conv(c1, c_, 1, 1, act=False)
    self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
    self.m = MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    self.n = n
    self.add = shortcut and c1 == c2

  def __call__(self, x:Tensor)->Tensor:
    y = [self.cv1(x)]
    y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
    y = self.cv2(y[0].cat(*y[1:], dim=1))
    return x + y if self.add else y

class Attention:
  def __init__(self, dim:int, num_heads:int=8, attn_ratio:float=0.5):
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.key_dim = int(self.head_dim * attn_ratio)
    self.scale = self.key_dim ** -0.5
    nh_kd = self.key_dim * num_heads
    h = dim + nh_kd * 2
    self.qkv = Conv(dim, h, 1, act=False)
    self.proj = Conv(dim, dim, 1, act=False)
    self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

  def __call__(self, x:Tensor)->Tensor:
    b, c, h, w = x.shape
    q,k,v= self.qkv(x).reshape(b, self.num_heads, self.key_dim * 2 + self.head_dim, h*w).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
    attn: Tensor = q*self.scale
    attn = attn.transpose(-2,-1) @ k
    attn = attn.softmax(axis=-1)
    x = (v @ attn.transpose(-2, -1)).reshape(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
    return self.proj(x)

class PSABlock:
  def __init__(self, c:int, attn_ratio:float=0.5, num_heads:int=4, shortcut:bool=True):
    self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
    self.ffn = [Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False)]
    self.add = shortcut

  def __call__(self, x:Tensor)->Tensor:
    x = x + self.attn(x) if self.add else self.attn(x)
    y = x
    for layer in self.ffn:
      y = layer(y)
    return x + y if self.add else y

class C2PSA:
  def __init__(self, c1, c2, n=1, e=0.5):
    assert c1 == c2
    self.c = int(c1 * e)
    self.cv1 = Conv(c1, 2 * self.c, 1, 1)
    self.cv2 = Conv(2 * self.c, c1, 1, 1)
    self.m = [PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64)  for _ in range(n)]

  def __call__(self, x):
    a, b = self.cv1(x).split((self.c, self.c), dim=1)
    for block in self.m:
      b = block(b)
    return self.cv2(a.cat(b, dim=1))

class Upsample:
  def __init__(self, scale_factor:int, mode: str = "nearest") -> None:
    assert mode == "nearest" # only mode supported for now
    self.mode = mode
    self.scale_factor = scale_factor

  def __call__(self, x: Tensor) -> Tensor:
    assert len(x.shape) > 2 and len(x.shape) <= 5
    (b, c), _lens = x.shape[:2], len(x.shape[2:])
    tmp = x.reshape([b, c, -1] + [1] * _lens) * Tensor.ones(*[1, 1, 1] + [self.scale_factor] * _lens)
    return tmp.reshape(list(x.shape) + [self.scale_factor] * _lens).permute([0, 1] + list(chain.from_iterable([[y+2, y+2+_lens] for y in range(_lens)]))).reshape([b, c] + [x * self.scale_factor for x in x.shape[2:]])

# utility functions for forward pass.
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
  lt, rb = distance.chunk(2, dim)
  x1y1 = anchor_points - lt
  x2y2 = anchor_points + rb
  if xywh:
    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return c_xy.cat(wh, dim=1)
  return x1y1.cat(x2y2, dim=1)

def make_anchors(feats, strides, grid_cell_offset=0.5):
  anchor_points, stride_tensor = [], []
  assert feats is not None
  for i, stride in enumerate(strides):
    _, _, h, w = feats[i].shape
    sx = Tensor.arange(w) + grid_cell_offset
    sy = Tensor.arange(h) + grid_cell_offset

    # this is np.meshgrid but in tinygrad
    sx = sx.reshape(1, -1).repeat([h, 1]).reshape(-1)
    sy = sy.reshape(-1, 1).repeat([1, w]).reshape(-1)

    anchor_points.append(Tensor.stack(sx, sy, dim=-1).reshape(-1, 2))
    stride_tensor.append(Tensor.full((h * w), stride))
  anchor_points = anchor_points[0].cat(anchor_points[1], anchor_points[2])
  stride_tensor = stride_tensor[0].cat(stride_tensor[1], stride_tensor[2]).unsqueeze(1)
  return anchor_points, stride_tensor

class DwConv(Conv):
  def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
    super().__init__(c1,c2,k,s,g=math.gcd(c1,c2), d=d, act=act)

class DFL():
  def __init__(self, c1:int=16):
    self.conv = Conv2d(c1, 1, 1, bias=False)
    x = Tensor.arange(c1, dtype='float')
    self.conv.weight.replace(x.reshape(1,c1,1,1))
    self.c1 = c1
  
  def __call__(self, x: Tensor) -> Tensor:
    b, _ ,a = x.shape
    return self.conv(x.view(b, 4, self.c1, a).transpose(2,1).softmax(1)).view(b, 4, a)

class Identity:
  def __call__(self, x): return x

class DetectionHead:
  dynamic = False  # force grid reconstruction
  export = False  # export mode
  format = None  # export format
  max_det = 300  # max_det
  agnostic_nms = False
  shape = None
  anchors = Tensor.empty(0)  # init
  strides = Tensor.empty(0)  # init
  xyxy = False  # xyxy or xywh output


  def __init__(self, nc=80, reg_max:int=16, end2end=False, ch=()):
    self.nc = nc  # number of classes
    self.nl = len(ch)
    self.reg_max = reg_max
    self.no = nc + self.reg_max * 4  #
    self.end2end = end2end
    self.stride = (8,16,32)
    c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100)) # channels
    self.cv2 = [
      Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), Conv2d(c2, 4*self.reg_max, 1)) for x in ch
    ]
    self.cv3 = [
      Sequential(
        Sequential(DwConv(x,x,3), Conv(x,c3,1)),
        Sequential(DwConv(c3,c3,3), Conv(c3,c3,1)),
        Conv2d(c3, self.nc, 1)
      )
      for x in ch
    ]
    self.dfl = DFL(self.reg_max) if self.reg_max > 1 else Identity()
    if end2end:
      self.one2one_cv2 = copy.deepcopy(self.cv2)
      self.one2one_cv3 = copy.deepcopy(self.cv3)

  @property
  def one2many(self):
    return dict(box_head=self.cv2, cls_head=self.cv3)

  @property
  def one2one(self):
    return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)
  
  @property
  def end2end(self):
    return getattr(self, "_end2end", False) and hasattr(self, "one2one_cv2") and hasattr(self, "one2one_cv3")

  @end2end.setter
  def end2end(self, value):
    self._end2end = value

  def _get_decode_boxes(self, x:dict[str,Tensor])->Tensor:
    shape = x["feats"][0].shape
    if self.dynamic or self.shape != shape:
      self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
      self.shape = shape

    dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
    return dbox

  def _inference(self, x: dict[str, Tensor]) -> Tensor:
    dbox = self._get_decode_boxes(x)
    return dbox.cat((x["scores"].sigmoid()), dim=1)

  def get_topk_index(self, scores: Tensor, max_det: int) -> tuple[Tensor, Tensor, Tensor]:
    batch_size, anchors, nc = scores.shape  # i.e. shape(16,8400,80)
    # Use max_det directly during export for TensorRT compatibility (requires k to be constant),
    # otherwise use min(max_det, anchors) for safety with small inputs during Python inference
    k = max_det if self.export else min(max_det, anchors)
    if self.agnostic_nms:
      scores, labels = scores.max(dim=-1, keepdim=True)
      scores, indices = scores.topk(k, dim=1)
      labels = labels.gather(1, indices)
      return scores, labels, indices
    ori_index = scores.max(axis=-1).topk(k)[1].unsqueeze(-1)
    scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
    scores, index = scores.flatten(1).topk(k)
    idx = ori_index[Tensor.arange(batch_size)[..., None], index // nc]  # original index
    return scores[..., None], (index % nc)[..., None].float(), idx

  def postprocess(self, preds: Tensor) -> Tensor:
    boxes, scores = preds.split([4, self.nc], dim=-1)
    conf, cls, idx = self.get_topk_index(scores, self.max_det)
    idx = idx.unsqueeze(-1)
    boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
    return boxes.cat(conf, cls, dim=-1)

  def decode_bboxes(self, bboxes: Tensor, anchors: Tensor, xywh: bool = True) -> Tensor:
    return dist2bbox(
        bboxes,
        anchors,
        xywh=xywh and not self.end2end and not self.xyxy,
        dim=1,
    )
  
  def forward_head(self, x:list[Tensor], box_head = None, cls_head = None):
    if box_head is None or cls_head is None:
      return dict()
    bs = x[0].shape[0]
    boxes = []
    for i in range(self.nl):
      boxes.append(box_head[i](x[i]).reshape(bs, 4 * self.reg_max, -1))
    boxes = boxes[0].cat(*boxes[1:], dim=2)
    scores = []
    for i in range(self.nl):
      scores.append(cls_head[i](x[i]).reshape(bs, self.nc, -1))
    scores = scores[0].cat(*scores[1:], dim=2)
    return dict(boxes=boxes, scores=scores, feats=x)
  
  def __call__(self, x:list[Tensor]):
    if self.end2end:
      preds = self.forward_head(x, **self.one2one)
      y = self._inference(preds)
      y = self.postprocess(y.permute(0,2,1))
      return y
    preds = self.forward_head(x, **self.one2many)
    y = self._inference(preds)
    return y

class Concat:
  def __init__(self, dim:int=1):
    self.dim = dim
  def __call__(self, x:Tensor):
    return x[0].cat(*x[1:], dim=self.dim)

class Sequential(list):
  __slots__ = ()

  def __init__(self, *layers):
    super().__init__(layers)

  def __call__(self, x):
    for layer in self:
      x = layer(x)
    return x

class YOLOv26:
  def __init__(self, w: float, d: float, ch: int, num_classes:int):
    c3k = w >= 1.0
    c64, c128, c256, c512, c1024 = (int(min(c, ch) * w) for c in (64, 128, 256, 512, 1024))
    self.model = [
      Conv(c1=3, c2=c64, k=3, s=2), # 0-P1/2 [-1, 1, Conv, [64, 3, 2]]
      Conv(c1=c64, c2=c128, k=3, s=2), # 1-P2/4 [-1, 1, Conv, [128, 3, 2]]
      C3K2(c1=c128,c2=c256,n=depth_scale(2,d),c3k=c3k,e=0.25,k=3), # [-1, 2, C3k2, [256, False, 0.25]]
      Conv(c1=c256, c2=c256, k=3, s=2), # 3-P3/8 [-1, 1, Conv, [256, 3, 2]]
      C3K2(c1=c256,c2=c512,n=depth_scale(2,d),c3k=c3k,e=0.25,k=3), # [-1, 2, C3k2, [512, False, 0.25]]
      Conv(c1=c512, c2=c512, k=3, s=2), # 5-P4/16 [-1, 1, Conv, [512, 3, 2]] 
      C3K2(c1=c512,c2=c512,c3k=True,n=depth_scale(2,d),k=3), # [-1, 2, C3k2, [512, True]]
      Conv(c1=c512, c2=c1024, k=3, s=2), # 7-P5/32 [-1, 1, Conv, [1024, 3, 2]]
      C3K2(c1=c1024,c2=c1024,c3k=True,n=depth_scale(2,d),k=3), # [-1, 2, C3k2, [512, True]]
      SPPF(c1=c1024,c2=c1024,k=5, n=3, shortcut=True), # [-1, 1, SPPF, [1024, 5, 3, True]] # 9
      C2PSA(c1=c1024,c2=c1024,n=depth_scale(2,d)), # [-1, 2, C2PSA, [1024]] # 10
      Upsample(scale_factor=2, mode="nearest"),
      Concat(),
      C3K2(c1=c1024 + c512, c2=c512, n=depth_scale(2, d), c3k=True, k=3),
      Upsample(scale_factor=2, mode="nearest"),
      Concat(),
      C3K2(c1=c512 + c512, c2=c256, n=depth_scale(2, d), c3k=True, k=3),
      Conv(c1=c256, c2=c256, k=3, s=2),
      Concat(),
      C3K2(c1=c256 + c512, c2=c512, n=depth_scale(2, d), c3k=True, k=3,),
      Conv(c1=c512, c2=c512, k=3, s=2),
      Concat(),
      C3K2(c1=c512 + c1024, c2=c1024, n=depth_scale(1, d), c3k=True, e=0.5, shortcut=True, k=3, attn=True),
      DetectionHead(num_classes, reg_max=1, end2end=True, ch=(c256, c512, c1024))
    ]


  def __call__(self, x: Tensor)->Tensor:
    outputs = []
    for i, layer in enumerate(self.model):
      if i == 12:
        x = layer([outputs[-1], outputs[6]])
      elif i == 15:
        x = layer([outputs[-1], outputs[4]])
      elif i == 18:
        x = layer([outputs[-1], outputs[13]])
      elif i == 21:
        x = layer([outputs[-1], outputs[10]])
      elif i == 23:
        x = layer([outputs[16], outputs[19], outputs[22]])
      else:
        x = layer(x)
      
      outputs.append(x)
    return x

def get_weights_location(yolo_variant: str) -> Path:
  def convert_f16_safetensor_to_f32(input_file: Path, output_file: Path):
    with open(input_file, 'rb') as f:
      metadata_length = int.from_bytes(f.read(8), 'little')
      metadata = json.loads(f.read(metadata_length).decode())
      float32_values = np.fromfile(f, dtype=np.float16).astype(np.float32)
      for v in metadata.values():
        if v["dtype"] == "F16": v.update({"dtype": "F32", "data_offsets": [offset * 2 for offset in v["data_offsets"]]})
      with open(output_file, 'wb') as f:
        new_metadata_bytes = json.dumps(metadata).encode()
        f.write(len(new_metadata_bytes).to_bytes(8, 'little'))
        f.write(new_metadata_bytes)
        float32_values.tofile(f)

  weights_location = Path(__file__).parents[1] / "weights" / f'yolo26{yolo_variant}.safetensors'
  fetch(f'https://huggingface.co/Acrusinho/yolo26-safetensors/resolve/main/yolo26{yolo_variant}.safetensors', weights_location)
  f32_weights = weights_location.with_name(f"{weights_location.stem}_f32.safetensors")
  if not f32_weights.exists(): convert_f16_safetensor_to_f32(weights_location, f32_weights)
  return f32_weights

def clip_boxes(boxes, shape):
  boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, shape[1])  # x1, x2
  boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, shape[0])  # y1, y2
  return boxes

def scale_boxes(img1_shape, predictions:np.ndarray, img0_shape, ratio_pad=None)->np.ndarray:
  gain = ratio_pad if ratio_pad else min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
  pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
  for pred in predictions:
    boxes_np = pred[:4].numpy() if isinstance(pred[:4], Tensor) else pred[:4]
    boxes_np[..., [0, 2]] -= pad[0]
    boxes_np[..., [1, 3]] -= pad[1]
    boxes_np[..., :4] /= gain
    boxes_np = clip_boxes(boxes_np, img0_shape)
    pred[:4] = boxes_np
  return predictions

def draw_bounding_boxes_and_save(orig_img_path, output_img_path, predictions, class_labels):
  color_dict = {label: tuple((((i+1) * 50) % 256, ((i+1) * 100) % 256, ((i+1) * 150) % 256)) for i, label in enumerate(class_labels)}
  font = cv2.FONT_HERSHEY_SIMPLEX

  def is_bright_color(color):
    r, g, b = color
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness > 127

  orig_img = cv2.imread(orig_img_path) if not isinstance(orig_img_path, np.ndarray) else cv2.imdecode(orig_img_path, 1)
  height, width, _ = orig_img.shape
  box_thickness = int((height + width) / 400)
  font_scale = (height + width) / 2500
  object_count = defaultdict(int)

  for pred in predictions:
    x1, y1, x2, y2, conf, class_id = pred
    if conf <= 0: continue
    x1, y1, x2, y2, class_id = map(int, (x1, y1, x2, y2, class_id))
    color = color_dict[class_labels[class_id]]
    cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, box_thickness)
    label = f"{class_labels[class_id]} {conf:.2f}"
    text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
    label_y, bg_y = (y1 - 4, y1 - text_size[1] - 4) if y1 - text_size[1] - 4 > 0 else (y1 + text_size[1], y1)
    cv2.rectangle(orig_img, (x1, bg_y), (x1 + text_size[0], bg_y + text_size[1]), color, -1)
    font_color = (0, 0, 0) if is_bright_color(color) else (255, 255, 255)
    cv2.putText(orig_img, label, (x1, label_y), font, font_scale, font_color, 1, cv2.LINE_AA)
    object_count[class_labels[class_id]] += 1

  print("Objects detected:")
  for obj, count in object_count.items():
    print(f"- {obj}: {count}")

  cv2.imwrite(output_img_path, orig_img)
  print(f'saved detections at {output_img_path}')

if __name__=="__main__":
  print(f"sys.args: {sys.argv}")
  if len(sys.argv)<2:
    print("Error: image path is required")
    sys.exit(1)
  img_path = sys.argv[1]
  yolo_variant = sys.argv[2] if len(sys.argv)>=3 else (print("No variant given, so choosing 'n' as the default. Yolo26 has different variants, you can choose from ['n', 's', 'm', 'l', 'x']") or 'n') # defaults to nano
  conf_thres = float(sys.argv[3]) if len(sys.argv)>=4 else 0.25
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
  state_dict = safe_load(get_weights_location(yolo_variant))
  load_state_dict(yolo_infer, state_dict)
  st = time()
  print(f"runnning inference")
  predictions = yolo_infer(preprocessed_image)
  predictions = predictions.numpy()[0]
  predictions = predictions[predictions[:,4] > conf_thres]
  print(f'did inference in {int(round(((time() - st) * 1000)))}ms')
  print(f"predictions {predictions}")
  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_text().split("\n")
  print(f"detections after confidence filter: {len(predictions)}")
  predictions = scale_boxes(preprocessed_image.shape[2:], predictions, image.shape)
  draw_bounding_boxes_and_save(orig_img_path=image_location, output_img_path=out_path, predictions=predictions, class_labels=class_labels)
  sys.exit(0)
