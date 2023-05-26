import tinygrad as tg
from tinygrad.nn import Conv2d,BatchNorm2d


class SPPF:
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        self.c1 = c1
        self.c2 = c2
        self.k = k

        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv2d(c1, c_, 1, 1)
        self.cv2 = Conv2d(c_ * 4, c2, 1, 1)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        x2 = x.pad2d(self.k // 2).max_pool2d(kernel_size=(5,5), stride=(1,1))
        x3 = x.pad2d(self.k // 2).max_pool2d(kernel_size=(5,5), stride=(1,1))
        x4 = x.pad2d(self.k // 2).max_pool2d(kernel_size=(5,5), stride=(1,1))
        concatenated = x.cat((x, x2, x3, x4), axis=1)
        return self.cv2(concatenated)
    