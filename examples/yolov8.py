from tinygrad.nn import Conv2d,BatchNorm2d
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d,BatchNorm2d
from tinygrad.helpers import dtypes
import numpy as np
import math
# Model architecture from https://github.com/ultralytics/ultralytics/issues/189


# UTIL FUNCTIONS

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
  lt, rb = distance.chunk(2, dim)
  x1y1 = anchor_points - lt
  x2y2 = anchor_points + rb
  if xywh:
    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return c_xy.cat(wh, dim=1)  # xywh bbox
  return x1y1.cat(x2y2, dim=1) # xyxy bbox

def make_anchors(feats, strides, grid_cell_offset=0.5):
  anchor_points, stride_tensor = [], []
  assert feats is not None
  for i, stride in enumerate(strides):
    _, _, h, w = feats[i].shape
    sx = np.arange(w) + grid_cell_offset  # shift x
    sy = np.arange(h) + grid_cell_offset  # shift y
    sy, sx = np.meshgrid(sy, sx, indexing='ij')
    anchor_points.append(np.stack((sx, sy), -1).reshape(-1, 2))
    stride_tensor.append(np.full((h * w, 1), stride))
  return np.concatenate(anchor_points), np.concatenate(stride_tensor)

# this function is from the original implementation
def autopad(k, p=None, d=1):  # kernel, padding, dilation
  if d > 1:
      k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
  if p is None:
      p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
  return p


# MODULE Definitions
class SPPF:
  def __init__(self, c1, c2, k=5):
    c_ = c1 // 2  # hidden channels
    self.cv1 = Conv_Block(c1, c_, k, 1)
    self.cv2 = Conv_Block(c_ * 4, c2, k, 1)
    self.maxpool = lambda x : x.pad2d((k // 2, k // 2, k // 2, k // 2)).max_pool2d(kernel_size=5, stride=1)
        
  def forward(self, x):
    x = self.cv1(x)
    x2 = self.maxpool(x)
    x3 = self.maxpool(x2)
    x4 = self.maxpool(x3)
    return self.cv2(x.cat(x2, x3, x4, dim=1))

class Conv_Block:
  def __init__(self, c1, c2, kernel_size=1, stride=1, groups=1, dilation=1, padding=None):
    self.conv = Conv2d(c1,c2, kernel_size, stride, padding= autopad(kernel_size, padding, dilation),bias=False, groups=groups, dilation=dilation)
    self.batch = BatchNorm2d(c2)

  def __call__(self, x):
    return self.conv(x).silu()
  
  
class Bottleneck:
  def __init__(self, c1, c2 , shortcut: bool, g=1, kernels: list = (3,3), channel_factor=0.5):
    c_ = int(c2 * channel_factor)
    self.cv1 = Conv_Block(c1, c_, kernel_size=kernels[0], stride=1, padding=None)
    self.cv2 = Conv_Block(c_, c2, kernel_size=kernels[1], stride=1, padding=None, groups=g)
    self.residual = c1 == c2 and shortcut
    
  def forward(self, x):
    return x + self.cv2(self.cv1(x)) if self.residual else self.cv2(self.cv1(x))


class C2f:
  def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
    self.c = int(c2 * e)  # hidden channels
    self.cv1 = Conv_Block(c1, 2 * self.c, 1, 1)
    self.cv2 = Conv_Block((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
    self.bottleneck = [Bottleneck(self.c, self.c, shortcut, g, k=[(3, 3), (3, 3)], e=1.0) for _ in range(n)]

  def forward(self, x):
    y = self.cv1(x)
    # TODO: maybe can use 'Tensor.chunck' here
    y_chunks = [y[:, :self.c], y[:, self.c:]]
    y2 = [m(y_chunks[-1]) for m in self.bottleneck]
    concatenated = tuple([y] + y_chunks + y2)
    return self.cv2(concatenated)

# TODO: test this. for some reason DFL isn't working with 3d tensors inputs, and it's exactly how it should work.
class DFL():
  def __init__(self, c1=16):
    self.conv = Conv2d(c1, 1, 1, bias=False)
    x = Tensor.arange(c1, dtypes.float32)
    self.conv.weight = x.reshape(1, c1, 1, 1)
    self.c1 = c1

  def forward(self, x):
    b, c, a = x.shape # batch, channels, anchors
    return self.conv(x.reshape(b, 4, self.c1, a).transpose(2, 1).softmax(1)).reshape(b, 4, a)

  
# incomplete and untested
class DetectionHead():
  anchors = Tensor.empty(0)
  strides = Tensor.empty(0)

  def __init__(self, nc=80, filters=()):
    super().__init__()
    self.ch = 16  # DFL channels
    self.nc = nc  # number of classes
    self.nl = len(filters)  # number of detection layers
    self.no = nc + self.ch * 4  # number of outputs per anchor
    self.stride = Tensor.zeros(self.nl)  # strides computed during build #TODO - figure this out

    c1 = math.max(filters[0], self.nc)
    c2 = math.max((filters[0] // 4, self.ch * 4))

    self.dfl = DFL(self.ch) #FIX this
    self.cls = [[Conv_Block(x, c1, 3), Conv_Block(c1, c1, 3), Conv2d(c1, self.nc, 1)] for x in filters]
    self.box = [[Conv_Block(x, c2, 3), Conv_Block(c2, c2, 3), Conv2d(c2, 4 * self.ch, 1)] for x in filters]
    
  def forward(self, x):
    for i in range(self.nl):
      x[i] = self.box[i](x[i]).cat(self.cls[i](x[i]), dim=1)
    self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
    x = Tensor.stack([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
    box, cls = x.split((self.ch * 4, self.nc), 1)
    a, b = self.dfl(box).chunk(2,1)
    a = self.anchors.unsqueeze(0) - a
    b = self.anchors.unsqueeze(0) + b
    box = Tensor.stack(((a + b) / 2, b - a), dim=1)
    return Tensor.stack((box * self.strides, cls.sigmoid()), dim=1)



      
    
    


