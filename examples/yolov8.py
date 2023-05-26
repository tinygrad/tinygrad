from tinygrad.nn import Conv2d,BatchNorm2d
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d,BatchNorm2d
from itertools import chain


# Model architecture from https://github.com/ultralytics/ultralytics/issues/189

class SPPF:
    def __init__(self, c1, c2, k=5):
        c_ = c1 // 2  # hidden channels
        k = k // 2
        self.cv1 = Conv2d(c1, c_, 1, 1)
        self.cv2 = Conv2d(c_ * 4, c2, 1, 1)
        self.maxpool = lambda x : x.pad2d((k, k, k, k)).max_pool2d(kernel_size=(5,5), stride=(1,1))
        
    def forward(self, x):
        x = self.cv1(x)
        x2 = self.maxpool(x)
        x3 = self.maxpool(x2)
        x4 = self.maxpool(x3)
        concatenated = x.cat((x2, x3, x4), dim=1)
        return self.cv2(concatenated)
    
class Conv_Block:
  def __init__(self, c1, c2, kernel_size, stride, padding, dilation=1, groups=1):
    self.conv = Conv2d(c1,c2, kernel_size, stride, padding, dilation=dilation, groups=groups,bias=False)
    self.batch = BatchNorm2d(c2)

  def __call__(self, x):
    return (self.batch(self.conv(x))).silu()
    
class Bottleneck:
  def __init__(self, c1, c2 , shortcut: bool, kernels: list, channel_factor):
    c_ = c2 * channel_factor
    self.cv1 = Conv_Block(c1, c_, kernel_size=kernels[0], stride=1, padding=1)
    self.cv2 = Conv_Block(c_, c2, kernel_size=kernels[1], stride=1, padding=1)
    self.residual = c1 == c2 and shortcut
    
  def __call__(self, x):
    return x + self.cv2(self.cv1(x)) if self.residual else self.cv2(self.cv1)


# FROM https://github.com/geohot/tinygrad/pull/784 by dc-dc-dc
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

class C2f:
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            self.c = int(c2 * e)  # hidden channels
            self.cv1 = Conv_Block(c1, 2 * self.c, 1, 1)
            self.cv2 = Conv_Block((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
            self.bottleneck = [Bottleneck(self.c, self.c, shortcut, g, k=[(3, 3), (3, 3)], e=1.0) for _ in range(n)]

    def forward(self, x):
            """Forward pass through C2f layer."""
            y = self.cv1(x)
            y_chunks = [y[:, :self.c], y[:, self.c:]]
            y2 = [m(y_chunks[-1]) for m in self.bottleneck]
            concatenated = tuple([y] + y_chunks + y2)
            return self.cv2(concatenated)

