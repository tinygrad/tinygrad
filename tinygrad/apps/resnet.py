# classification in 50 lines
import sys
from tinygrad import nn, Tensor

class Bottleneck:
  expansion = 4
  def __init__(self, in_c, mid_c, stride=1):
    out_c = mid_c * self.expansion
    self.conv1, self.bn1 = nn.Conv2d(in_c, mid_c, 1, bias=False), nn.BatchNorm2d(mid_c)
    self.conv2, self.bn2 = nn.Conv2d(mid_c, mid_c, 3, stride, 1, bias=False), nn.BatchNorm2d(mid_c)
    self.conv3, self.bn3 = nn.Conv2d(mid_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c)
    self.downsample = (stride != 1 or in_c != out_c) and [nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c)] or []

  def __call__(self, x:Tensor) -> Tensor:
    id = x.sequential(self.downsample)
    x = self.bn1(self.conv1(x)).relu()
    x = self.bn2(self.conv2(x)).relu()
    x = self.bn3(self.conv3(x))
    return (x + id).relu()

class ResNet50:
  def __init__(self, num_classes=1000):
    self.conv1, self.bn1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(64,  64,  3, 1)
    self.layer2 = self._make_layer(256, 128, 4, 2)
    self.layer3 = self._make_layer(512, 256, 6, 2)
    self.layer4 = self._make_layer(1024,512, 3, 2)
    self.fc = nn.Linear(2048, num_classes)

  def _make_layer(self, in_c, mid_c, blocks, stride):
    layers = [Bottleneck(in_c, mid_c, stride)]
    for _ in range(1, blocks): layers.append(Bottleneck(mid_c * Bottleneck.expansion, mid_c))
    return layers

  def __call__(self, x:Tensor) -> Tensor:
    x = self.bn1(self.conv1(x)).relu()
    x = x.max_pool2d()
    x = x.sequential([self.layer1, self.layer2, self.layer3, self.layer4])
    x = x.mean((2, 3))
    return self.fc(x)

if __name__ == "__main__":
  state_dict = nn.state.safe_load(Tensor.from_url("https://huggingface.co/timm/resnet50.a1_in1k/resolve/main/model.safetensors"))
  model = ResNet50()
  nn.state.load_state_dict(model, state_dict)
  img = nn.state.png_load(Tensor.from_url(sys.argv[1]))
  print(model(img).argmax())