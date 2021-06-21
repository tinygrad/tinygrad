from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from extra.utils import fetch, fake_torch_load
from torch.hub import load_state_dict_from_url
import numpy as np

model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def load_from_pretrained(model, url):
  state_dict = load_state_dict_from_url(url, progress=True)
  layers_not_loaded = []
  for k, v in state_dict.items():
    par_name = ['model']
    for kk in k.split('.'):
      if kk.isdigit():
        par_name += [f'layers[{int(kk)}]']
      else:
        par_name += [kk]
    par_name = '.'.join(par_name)
    code = f"""
if np.prod({par_name}.shape) == np.prod(v.shape):\n
  if "fc.weight" in par_name:\n
    {par_name}.assign(Tensor(v.detach().numpy().T))\n
  else:\n
    {par_name}.assign(Tensor(v.detach().numpy()))\n
else:\n
  layers_not_loaded += [k]"""
    exec(code)
  print(f'Loaded from "{url}".')
  if len(layers_not_loaded) > 0:
    for l in layers_not_loaded:
      print(f'- Layer {l} not loaded.')
  return model

class BasicBlock:
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2D(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = nn.BatchNorm2D(planes)
    self.downsample = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.downsample = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2D(self.expansion*planes)
      )

  def __call__(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out))
    out = out + self.downsample(x)
    out = out.relu()
    return out


class Bottleneck:
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2D(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
    self.bn2 = nn.BatchNorm2D(planes)
    self.conv3 = nn.Conv2d(planes, self.expansion *planes, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2D(self.expansion*planes)
    self.downsample = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.downsample = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2D(self.expansion*planes)
      )

  def __call__(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out)).relu()
    out = self.bn3(self.conv3(out))
    out = out + self.downsample(x)
    out = out.relu()
    return out

class ResNet:
  def __init__(self, block, num_blocks, num_classes=10, pretrained=False):
    self.in_planes = 64

    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3)
    self.bn1 = nn.BatchNorm2D(64)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = out.mean(3).mean(2)
    out = self.fc(out).logsoftmax()
    return out

  def __call__(self, x):
    return self.forward(x)

def ResNet18(num_classes, pretrained=False):
  model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
  if pretrained:
    model = load_from_pretrained(model, model_urls['resnet18'])
  return model

def ResNet34(num_classes, pretrained=False):
  model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
  if pretrained:
    model = load_from_pretrained(model, model_urls['resnet34'])
  return model

def ResNet50(num_classes, pretrained=False):
  model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
  if pretrained:
    model = load_from_pretrained(model, model_urls['resnet50'])
  return model

def ResNet101(num_classes, pretrained=False):
  model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
  if pretrained:
    model = load_from_pretrained(model, model_urls['resnet101'])
  return model

def ResNet152(num_classes, pretrained=False):
  model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes, pretrained=pretrained)
  if pretrained:
    model = load_from_pretrained(model, model_urls['resnet152'])
  return model
