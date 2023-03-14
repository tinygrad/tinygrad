from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from extra.utils import get_child
import numpy as np

class BasicBlock:
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = []
    if stride != 1 or in_planes != self.expansion*planes:
      self.downsample = [
        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion*planes)
      ]

  def __call__(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out))
    out = out + x.sequential(self.downsample)
    out = out.relu()
    return out


class Bottleneck:
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, self.expansion *planes, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion*planes)
    self.downsample = []
    if stride != 1 or in_planes != self.expansion*planes:
      self.downsample = [
        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion*planes)
      ]

  def __call__(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out)).relu()
    out = self.bn3(self.conv3(out))
    out = out + x.sequential(self.downsample)
    out = out.relu()
    return out

class ResNet:
  # def __init__(self, block, num_blocks, num_classes=10, url=None):
  def __init__(self, num, num_classes):
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

    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=2)
    self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
    self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
    self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
    self.fc = {"weight": Tensor.uniform(512 * self.block.expansion, num_classes), "bias": Tensor.zeros(num_classes)}

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return layers

  def forward(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = out.sequential(self.layer1)
    out = out.sequential(self.layer2)
    out = out.sequential(self.layer3)
    out = out.sequential(self.layer4)
    out = out.mean(3).mean(2)
    out = out.linear(**self.fc).log_softmax()
    return out

  def __call__(self, x):
    return self.forward(x)

  def load_from_pretrained(self):
    # TODO replace with fake torch load
  
    model_urls = {
      18: 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
      34: 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
      50: 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
      101: 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
      152: 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    }

    self.url = model_urls[self.num]

    from torch.hub import load_state_dict_from_url
    state_dict = load_state_dict_from_url(self.url, progress=True)
    for k, v in state_dict.items():
      obj = get_child(self, k)
      dat = v.detach().numpy().T if "fc.weight" in k else v.detach().numpy()

      if 'fc.' in k and obj.shape != dat.shape:
        print("skipping fully connected layer")
        continue # Skip FC if transfer learning

      assert obj.shape == dat.shape, (k, obj.shape, dat.shape)
      obj.assign(dat)

ResNet18 = lambda num_classes=1000: ResNet(18, num_classes=num_classes)
ResNet34 = lambda num_classes=1000: ResNet(34, num_classes=num_classes)
ResNet50 = lambda num_classes=1000: ResNet(50, num_classes=num_classes)
ResNet101 = lambda num_classes=1000: ResNet(101, num_classes=num_classes)
ResNet152 = lambda num_classes=1000: ResNet(152, num_classes=num_classes)
