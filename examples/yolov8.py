from tinygrad.nn import Conv2d,BatchNorm2d
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d,BatchNorm2d
import numpy as np
import math
from itertools import chain
from extra.utils import download_file, get_child
from pathlib import Path
import torch


#Model architecture from https://github.com/ultralytics/ultralytics/issues/189
#the upsampling class has been taken from this pull request by dc-dc-dc. Now 2 models use upsampling. (retinet and this)


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
    sx = np.arange(w, dtype='float32') + grid_cell_offset  # shift x
    sy = np.arange(h, dtype='float32') + grid_cell_offset  # shift y
    sy, sx = np.meshgrid(sx, sy, indexing='ij')
    anchor_points.append(np.stack((sy, sx), -1).reshape(-1, 2))
    stride_tensor.append(np.full((h * w, 1), stride.cpu().numpy()))
  return np.concatenate(anchor_points).reshape(2, -1), np.concatenate(stride_tensor).reshape(1, -1)

# this function is from the original implementation
def autopad(k, p=None, d=1):  # kernel, padding, dilation
  if d > 1:
    k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
  if p is None:
    p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
  return p

#this is taken from https://github.com/geohot/tinygrad/pull/784/files by dc-dc-dc (Now 2 models use upsampling)
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
  
# MODULE Definitions
class SPPF:
  def __init__(self, c1, c2, k=5):
      c_ = c1 // 2  # hidden channels
      self.cv1 = Conv_Block(c1, c_, k, 1, padding=None)
      self.cv2 = Conv_Block(c_ * 4, c2, k, 1, padding=None)
      self.maxpool = lambda x : x.pad2d((k // 2, k // 2, k // 2, k // 2)).max_pool2d(kernel_size=k, stride=1)
        
  def __call__(self, x):
    x = self.cv1(x)
    x2 = self.maxpool(x)
    x3 = self.maxpool(x2)
    x4 = self.maxpool(x3)
    return self.cv2(x.cat(x2, x3, x4, dim=1))
      
class Conv_Block:
  def __init__(self, c1, c2, kernel_size=1, stride=1, groups=1, dilation=1, padding=None):
    self.conv = Conv2d(c1,c2, kernel_size, stride, padding= autopad(kernel_size, padding, dilation), bias=False, groups=groups, dilation=dilation)
    self.bn = BatchNorm2d(c2)

  def __call__(self, x):
    return self.bn(self.conv(x)).silu()
   
class Bottleneck:
  def __init__(self, c1, c2 , shortcut: bool, g=1, kernels: list = (3,3), channel_factor=0.5):
    c_ = int(c2 * channel_factor)
    self.cv1 = Conv_Block(c1, c_, kernel_size=kernels[0], stride=1, padding=None)
    self.cv2 = Conv_Block(c_, c2, kernel_size=kernels[1], stride=1, padding=None, groups=g)
    self.residual = c1 == c2 and shortcut
    
  def __call__(self, x):
    return x + self.cv2(self.cv1(x)) if self.residual else self.cv2(self.cv1(x))
                  
class C2f:
  def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
    self.c = int(c2 * e)  # hidden channels
    self.cv1 = Conv_Block(c1, 2 * self.c, 1,)
    self.cv2 = Conv_Block((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
    self.bottleneck = [Bottleneck(self.c, self.c, shortcut, g, kernels=[(3, 3), (3, 3)], channel_factor=1.0) for _ in range(n)]
   
  def __call__(self, x):
    y= list(self.cv1(x).chunk(2, 1))
    y.extend(m(y[-1]) for m in self.bottleneck)
    z = y[0]
    for i in y[1:]: z = z.cat(i, dim=1)
    return self.cv2(z)

class DFL():
  def __init__(self, c1=16):
    self.conv = Conv2d(c1, 1, 1, bias=False)
    x = Tensor.arange(c1)
    self.conv.weight.assign(x.reshape(1, c1, 1, 1))
    self.c1 = c1

  def __call__(self, x):
    b, c, a = x.shape # batch, channels, anchors
    return self.conv(x.reshape(b, 4, self.c1, a).transpose(2, 1).softmax(1)).reshape(b, 4, a)


# stride = tensor([ 8., 16., 32.])
class DetectionHead():
  def __init__(self, nc=80, filters=()):
    self.ch = 16  # DFL channels
    self.nc = nc  # number of classes
    self.nl = len(filters)  # number of detection layers
    self.no = nc + self.ch * 4  # number of outputs per anchor
    self.stride = Tensor([8, 16, 32])
    c1 = max(filters[0], self.nc)
    c2 = max((filters[0] // 4, self.ch * 4))

    self.dfl = DFL(self.ch) 
    self.cv3 = [[Conv_Block(x, c1, 3), Conv_Block(c1, c1, 3), Conv2d(c1, self.nc, 1)] for x in filters]
    self.cv2 = [[Conv_Block(x, c2, 3), Conv_Block(c2, c2, 3), Conv2d(c2, 4 * self.ch, 1)] for x in filters]
  
  def forward(self, x):
    for i in range(self.nl):
      x[i] = x[i].sequential(self.cv2[i]).cat(x[i].sequential(self.cv3[i]), dim=1)
    self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
    y = [i.reshape(x[0].shape[0], self.no, -1) for i in x]
    x_cat = y[0].cat(y[1], y[2], dim=2)
    split_sizes = [self.ch * 4, self.nc]
    box, cls = [x_cat[:, :split_sizes[0], :], x_cat[:, split_sizes[0]:, :]]
    dbox = dist2bbox(self.dfl(box), Tensor(self.anchors).unsqueeze(0), xywh=True, dim=1) * Tensor(self.strides)
    z = dbox.cat(cls.sigmoid(), dim=1)
    return z
             
                           
class Darknet():
  def __init__(self, w, r, d): #width_multiple, ratio_multiple, depth_multiple
    self.b1 = [Conv_Block(c1=3, c2= int(64*w), kernel_size=3, stride=2, padding=1), Conv_Block(int(64*w), int(128*w), kernel_size=3, stride=2, padding=1)]
    self.b2 = [C2f(c1=int(128*w), c2=int(128*w), n=round(3*d), shortcut=True), Conv_Block(int(128*w), int(256*w), 3, 2, 1), C2f(int(256*w), int(256*w), round(6*d), True)]
    self.b3 = [Conv_Block(int(256*w), int(512*w), kernel_size=3, stride=2, padding=1), C2f(int(512*w), int(512*w), round(6*d), True)]
    self.b4 = [Conv_Block(int(512*w), int(512*w*r), kernel_size=3, stride=2, padding=1), C2f(int(512*w*r), int(512*w*r), round(3*d), True)]
    self.b5 = [SPPF(int(512*w*r), int(512*w*r), 1)]
    
  def return_modules(self):
    return [*self.b1, *self.b2, *self.b3, *self.b4, *self.b5]
  
  def forward(self, x):
    x1 = x.sequential(self.b1)
    x2 = x1.sequential(self.b2)
    x3 = x2.sequential(self.b3)
    x4 = x3.sequential(self.b4)
    x5 = self.b5[0](x4)
    return (x2, x3, x5)
  
class Yolov8NECK():
  def __init__(self, w, r, d):  #width_multiple, ratio_multiple, depth_multiple
    self.up = Upsample(2, mode='nearest')
    self.n1 = C2f(c1=int(512*w*(1+r)), c2=int(512*w), n=round(3*d), shortcut=False)
    self.n2 = C2f(c1=int(768*w), c2=int(256*w), n=round(3*d), shortcut=False)
    self.n3 = Conv_Block(c1=int(256*w), c2=int(256*w), kernel_size=3, stride=2, padding=1)
    self.n4 = C2f(c1=int(768*w), c2=int(512*w), n=round(3*d), shortcut=False)
    self.n5 = Conv_Block(c1=int(512* w), c2=int(512 * w), kernel_size=3, stride=2, padding=1)
    self.n6 = C2f(c1=int(512*w*(1+r)), c2=int(512*w*r), n=round(3*d), shortcut=False)
  
  def return_modules(self):
    return [self.n1, self.n2, self.n3, self.n4, self.n5, self.n6]
  
  def forward(self, p3, p4, p5):
    x =  self.n1(p4.cat(self.up(p5), dim=1))
    head_1 = self.n2(p3.cat(self.up(x), dim=1))
    head_2 = self.n4(x.cat(self.n3(head_1), dim=1))
    head_3 = self.n6(p5.cat(self.n5(head_2), dim=1))
    return [head_1, head_2, head_3]

class YOLOv8():
  # confirm filters. 
  def __init__(self, w, r,  d, num_classes): #width_multiple, ratio_multiple, depth_multiple
    self.net = Darknet(w, r, d)
    self.fpn = Yolov8NECK(w, r, d)
    self.head = DetectionHead(num_classes, filters=(int(256*w), int(512*w), int(512*w*r)))

  def forward(self, x):
    x = self.net.forward(x)
    x = self.fpn.forward(*x)
    return self.head.forward(x)

  def load_weights(self):
    weights_path = Path(__file__).parent.parent / "weights" / "yolov8s.pt"
    state_dict = torch.load(weights_path)
    weights = state_dict['model'].state_dict().items()
    backbone_modules = [*range(10)]
    yolov8neck_modules = [12, 15, 16, 18, 19, 21]
    yolov8_head_weights = [(22, self.head)]
    all_trainable_weights = [*zip(backbone_modules, self.net.return_modules()), *zip(yolov8neck_modules, self.fpn.return_modules()), *yolov8_head_weights]
    for k, v in weights:
      k = k.split('.')
      for i in all_trainable_weights:
        if int(k[1]) in i:
          child_key = '.'.join(k[2:]) if k[2] != 'm' else 'bottleneck.' + '.'.join(k[3:])
          get_child(i[1], child_key).assign(v.numpy())
    print('successfully loaded all weights')
    
       
test_inferece = Tensor.rand(1 ,3 , 640 , 640)
yolo_infer = YOLOv8(w=0.5, r=2, d=0.33, num_classes=80)  
print(yolo_infer.forward(test_inferece))
yolo_infer.load_weights()


# post processing functions for raw outputs from the head "https://github.com/ultralytics/ultralytics/blob/dada5b73c4340671ac67b99e8c813bf7b16c34ce/ultralytics/yolo/v8/detect/predict.py"
#Saving --> plotting function - write results. 
#pre_process --> image process

def clip_boxes(boxes, shape):
  boxes_np = boxes.numpy() if isinstance(boxes, Tensor) else boxes
  boxes_np[..., [0, 2]] = np.clip(boxes_np[..., [0, 2]], 0, shape[1])  # x1, x2
  boxes_np[..., [1, 3]] = np.clip(boxes_np[..., [1, 3]], 0, shape[0])  # y1, y2
  return Tensor(boxes_np) if isinstance(boxes, Tensor) else boxes_np

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
  gain, pad = ratio_pad if ratio_pad else (min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]), ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2))
  boxes_np = boxes.numpy() if isinstance(boxes, Tensor) else boxes
  boxes_np[..., [0, 2]] -= pad[0]
  boxes_np[..., [1, 3]] -= pad[1]
  boxes_np[..., :4] /= gain
  boxes_np = clip_boxes(boxes_np, img0_shape)
  return Tensor(boxes_np) if isinstance(boxes, Tensor) else boxes_np

def xywh2xyxy(x):
  x_np = x.numpy() if isinstance(x, Tensor) else x
  xy = x_np[..., :2]  # center x, y
  wh = x_np[..., 2:4]  # width, height
  xy1 = xy - wh / 2  # top left x, y
  xy2 = xy + wh / 2  # bottom right x, y
  result = np.concatenate((xy1, xy2), axis=-1)
  return Tensor(result) if isinstance(x, Tensor) else result

# TODO: mismatch 5% with pytorch
# def box_iou(box1, box2):
#   box1_np = box1.numpy() if isinstance(box1, Tensor) else box1
#   box2_np = box2.numpy() if isinstance(box2, Tensor) else box2
#   a1, a2 = np.split(box1_np[:, None], 2, axis=2)
#   b1, b2 = np.split(box2_np, 2, axis=1)
#   intersection = (np.minimum(a2, b2) - np.maximum(a1, b1)).clip(0).prod(2)
#   area1 = (box1_np[:, 2] - box1_np[:, 0]) * (box1_np[:, 3] - box1_np[:, 1])
#   area2 = (box2_np[:, 2] - box2_np[:, 0]) * (box2_np[:, 3] - box2_np[:, 1])
#   result = intersection / (area1[:, None] + area2 - intersection)
#   return Tensor(result) if isinstance(box1, Tensor) else result
    


