from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, LayerNorm, Linear

class Block:
  def __init__(self, dim):
    self.dwconv = Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
    self.norm = LayerNorm(dim, eps=1e-6)
    self.pwconv1 = Linear(dim, 4 * dim)
    self.pwconv2 = Linear(4 * dim, dim)
    self.gamma = Tensor.ones(dim)

  def __call__(self, x:Tensor):
    return x + x.sequential([
      self.dwconv, lambda x: x.permute(0, 2, 3, 1), self.norm,
      self.pwconv1, Tensor.gelu, self.pwconv2, lambda x: (self.gamma * x).permute(0, 3, 1, 2)
    ])

class ConvNeXt:
  def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
    self.downsample_layers = [
      [Conv2d(in_chans, dims[0], kernel_size=4, stride=4), LayerNorm((dims[0], 1, 1), eps=1e-6)],
      *[[LayerNorm((dims[i], 1, 1), eps=1e-6), Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)] for i in range(3)]
    ]
    self.stages = [[Block(dims[i]) for _ in range(depths[i])] for i in range(4)]
    self.norm = LayerNorm(dims[-1])
    self.head = Linear(dims[-1], num_classes)

  def __call__(self, x:Tensor):
    for downsample, stage in zip(self.downsample_layers, self.stages):
      x = x.sequential(downsample).sequential(stage)
    return x.mean([-2, -1]).sequential([self.norm, self.head])

if __name__ == "__main__":
  model = ConvNeXt()

  from extra.utils import fetch, fake_torch_load, get_child
  weights = fake_torch_load(fetch('https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth'))['model']
  for k,v in weights.items():
    mv = get_child(model, k)
    mv.assign(v.reshape(mv.shape)).realize()

  from test.models.test_efficientnet import chicken_img, preprocess, _LABELS
  img = Tensor(preprocess(chicken_img))

  Tensor.training = False
  Tensor.no_grad = True

  out = model(img).numpy()
  print(_LABELS[out.argmax()])
