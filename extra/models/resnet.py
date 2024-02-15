import math
from typing import List
import tinygrad.nn as nn
from tinygrad.tensor import Tensor
from tinygrad.nn.state import torch_load
from tinygrad.helpers import fetch, get_child, getenv, prod, argfix
from tinygrad.dtype import dtypes
from tinygrad.features.multi import MultiLazyBuffer

class UnsyncedBatchNorm:
  def __init__(self, num_features, num_devices=getenv("GPUS", 1)):
    self.bns:List[nn.BatchNorm2d] = []
    for _ in range(num_devices):
      bn = nn.BatchNorm2d(num_features)
      self.bns.append(bn)

  def __call__(self, x:Tensor):
    if len(self.bns) == 1: return self.bns[0](x)

    bn_ts = []
    assert isinstance(x.lazydata, MultiLazyBuffer)
    for bound, bn in zip(x.lazydata.bounds, self.bns):
      # TODO: __getitem__ does not work
      # xi = x[bound]
      xi = x.shrink((bound, None, None, None))
      bni = bn(xi)
      bn_ts.append(bni)
    # TODO: what do we want to do for inference? average weight? pick any one?
    # a good start would be to check each mean/std are similar
    return bn_ts[0].cat(*bn_ts[1:])
  # todo: hack, this make loading from weights work on 1 gpu...
  def __getattr__(self, item): return getattr(self.bns[0], item)

BatchNorm = nn.BatchNorm2d if getenv("SYNCBN", 0) else UnsyncedBatchNorm


# rejection sampling truncated randn
def randn(*shape, dtype=None, truncstds=2, **kwargs) -> Tensor:
  CNT=8
  x = Tensor.randn(*(*shape, CNT), dtype=dtype, **kwargs)
  ctr = Tensor.arange(CNT).reshape((1,) * len(x.shape[:-1]) + (CNT,)).expand(x.shape)
  take = (x.abs() <= truncstds).where(ctr, CNT).min(axis=-1, keepdim=True)  # set to 0 if no good samples
  return (ctr == take).where(x, 0).sum(axis=-1)


class Conv2dHeNormal(nn.Conv2d):
  def initialize_weight(self, out_channels, in_channels, groups):
    # https://github.com/keras-team/keras/blob/v2.15.0/keras/initializers/initializers.py#L1026-L1065
    def he_normal(*shape, a: float = 0.00, **kwargs) -> Tensor:
      std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:])) / 0.87962566103423978
      return std * randn(*shape, **kwargs)
    return he_normal(out_channels, in_channels//groups, *self.kernel_size, a=0.0)

class Linear(nn.Linear):
  def initialize_weight(self, in_features, out_features):
    return Tensor.normal((out_features, in_features), mean=0.0, std=0.01)
  def initialize_bias(self, in_features, out_features):
    return Tensor.zeros(out_features)

class BasicBlock:
  expansion = 1

  def __init__(self, in_planes, planes, stride=1, groups=1, base_width=64):
    assert groups == 1 and base_width == 64, "BasicBlock only supports groups=1 and base_width=64"
    self.conv1 = Conv2dHeNormal(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = BatchNorm(planes)
    self.conv2 = Conv2dHeNormal(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = BatchNorm(planes)
    self.downsample = []
    self.bn_downsample = None
    if stride != 1 or in_planes != self.expansion*planes:
      self.bn_downsample = BatchNorm(self.expansion*planes)  # name this BN so LARS can exclude it
      self.downsample = [
        Conv2dHeNormal(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        self.bn_downsample
      ]

  def __call__(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out))
    out = out + x.sequential(self.downsample)
    out = out.relu()
    return out


class Bottleneck:
  # NOTE: stride_in_1x1=False, this is the v1.5 variant
  expansion = 4

  def __init__(self, in_planes, planes, stride=1, stride_in_1x1=False, groups=1, base_width=64):
    width = int(planes * (base_width / 64.0)) * groups
    # NOTE: the original implementation places stride at the first convolution (self.conv1), control with stride_in_1x1
    self.conv1 = Conv2dHeNormal(in_planes, width, kernel_size=1, stride=stride if stride_in_1x1 else 1, bias=False)
    self.bn1 = BatchNorm(width)
    self.conv2 = Conv2dHeNormal(width, width, kernel_size=3, padding=1, stride=1 if stride_in_1x1 else stride, groups=groups, bias=False)
    self.bn2 = BatchNorm(width)
    self.conv3 = Conv2dHeNormal(width, self.expansion*planes, kernel_size=1, bias=False)
    self.bn3 = BatchNorm(self.expansion*planes)
    self.downsample = []
    if stride != 1 or in_planes != self.expansion*planes:
      self.downsample = [
        Conv2dHeNormal(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        BatchNorm(self.expansion*planes)
      ]

  def __call__(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out)).relu()
    out = self.bn3(self.conv3(out))
    out = out + x.sequential(self.downsample)
    out = out.relu()
    return out

class ResNet:
  def __init__(self, num, num_classes=None, groups=1, width_per_group=64, stride_in_1x1=False):
    self.num = num
    self.block = {
      18: BasicBlock,
      34: BasicBlock,
      50: Bottleneck,
      101: Bottleneck,
      152: Bottleneck
    }[num]

    self.num_blocks = {
      18: [2,2,2,2],
      34: [3,4,6,3],
      50: [3,4,6,3],
      101: [3,4,23,3],
      152: [3,8,36,3]
    }[num]

    self.in_planes = 64

    self.groups = groups
    self.base_width = width_per_group
    self.conv1 = Conv2dHeNormal(3, 64, kernel_size=7, stride=2, bias=False, padding=3)
    self.bn1 = BatchNorm(64)
    self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1, stride_in_1x1=stride_in_1x1)
    self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2, stride_in_1x1=stride_in_1x1)
    self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2, stride_in_1x1=stride_in_1x1)
    self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2, stride_in_1x1=stride_in_1x1)
    self.fc = Linear(512 * self.block.expansion, num_classes) if num_classes is not None else None

  def _make_layer(self, block, planes, num_blocks, stride, stride_in_1x1):
    strides = [stride] + [1] * (num_blocks-1)
    layers = []
    for stride in strides:
      if block == Bottleneck:
        layers.append(block(self.in_planes, planes, stride, stride_in_1x1, self.groups, self.base_width))
      else:
        layers.append(block(self.in_planes, planes, stride, self.groups, self.base_width))
      self.in_planes = planes * block.expansion
    return layers

  def forward(self, x):
    is_feature_only = self.fc is None
    if is_feature_only: features = []
    out = self.bn1(self.conv1(x)).relu()
    out = out.pad2d([1,1,1,1]).max_pool2d((3,3), 2)
    out = out.sequential(self.layer1)
    if is_feature_only: features.append(out)
    out = out.sequential(self.layer2)
    if is_feature_only: features.append(out)
    out = out.sequential(self.layer3)
    if is_feature_only: features.append(out)
    out = out.sequential(self.layer4)
    if is_feature_only: features.append(out)
    if not is_feature_only:
      out = out.mean([2,3])
      out = self.fc(out)
      return out
    return features

  def __call__(self, x:Tensor) -> Tensor:
    return self.forward(x)

  def load_from_pretrained(self):
    # TODO replace with fake torch load

    model_urls = {
      (18, 1, 64): 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
      (34, 1, 64): 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
      (50, 1, 64): 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
      (50, 32, 4): 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
      (101, 1, 64): 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
      (152, 1, 64): 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    self.url = model_urls[(self.num, self.groups, self.base_width)]
    for k, v in torch_load(fetch(self.url)).items():
      obj: Tensor = get_child(self, k)
      dat = v.detach().numpy()

      if 'fc.' in k and obj.shape != dat.shape:
        print("skipping fully connected layer")
        continue # Skip FC if transfer learning

      # TODO: remove or when #777 is merged
      assert obj.shape == dat.shape or (obj.shape == (1,) and dat.shape == ()), (k, obj.shape, dat.shape)
      obj.assign(dat)

ResNet18 = lambda num_classes=1000: ResNet(18, num_classes=num_classes)
ResNet34 = lambda num_classes=1000: ResNet(34, num_classes=num_classes)
ResNet50 = lambda num_classes=1000: ResNet(50, num_classes=num_classes)
ResNet101 = lambda num_classes=1000: ResNet(101, num_classes=num_classes)
ResNet152 = lambda num_classes=1000: ResNet(152, num_classes=num_classes)
ResNeXt50_32X4D = lambda num_classes=1000: ResNet(50, num_classes=num_classes, groups=32, width_per_group=4)